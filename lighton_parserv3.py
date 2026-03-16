import os
import sys
import gc
import argparse
import torch
import fitz  # PyMuPDF
import re  # <--- ADDED FOR ANTI-LOOP CLEANUP
from PIL import Image
from tqdm import tqdm
# Updated to the official V2 classes
from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor


class LightOnOCRParser:
    """
    Standalone parser using LightOnOCR-2-1B.
    Follows official documentation for API routing and 1540px scaling.
    Includes aggressive anti-repetition parameters.
    """

    def __init__(self,
                 model_id="lightonai/LightOnOCR-2-1B",  # Upgraded to V2
                 output_dir="./output",
                 dpi=200,
                 max_new_tokens=6000):  # <--- INCREASED to 6000 to handle the heavy Hindi token tax

        self.model_id = model_id
        self.output_dir = output_dir
        self.dpi = dpi
        self.max_new_tokens = max_new_tokens

        self._load_model()

    def _load_model(self):
        print(f"Loading {self.model_id} in 16-bit (bfloat16) for 4GB VRAM...")

        # Using the specific V2 Processor
        self.processor = LightOnOcrProcessor.from_pretrained(self.model_id)

        self.model = LightOnOcrForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("Model loaded successfully!")

    def _inference(self, image):
        """Runs the actual AI inference using the official V2 template."""

        # Official Rendering Tip: "target longest dimension of 1540px"
        image.thumbnail((1540, 1540), Image.Resampling.LANCZOS)

        # V2 API: We pass the PIL image object directly into the conversation dict
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image}
                ]
            }
        ]

        # V2 API: tokenize=True processes the image and text together
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        # Cast tensors to bfloat16 and move to GPU
        inputs = {k: v.to(device="cuda", dtype=torch.bfloat16) if v.is_floating_point() else v.to("cuda") for k, v in
                  inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                use_cache=True,
                do_sample=False,
                repetition_penalty=1.25  # <--- INCREASED to strictly prevent Hindi character looping
            )

        # Decode tokens
        input_length = inputs["input_ids"].shape[1]
        generated_ids = output_ids[0, input_length:]
        extracted_text = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

        # --- NEW: ANTI-LOOP CLEANUP ---
        # If the model repeats a block of text 3 or more times at the end, chop it off.
        extracted_text = re.sub(r'(.{20,}?)(?:\s*\1){3,}', r'\1', extracted_text, flags=re.DOTALL)

        # --- NEW: MARGINALIA & LATEX ARTIFACT CLEANUP ---
        # 1. Removes bold LaTeX variables/numbers (e.g., $\mathbf{45}$)
        extracted_text = re.sub(r'\$\\mathbf\{[^}]+\}\$\.?\n?', '', extracted_text)

        # 2. Removes LaTeX text tags containing Hindi words/numbers (e.g., $\text{५}$, $\text{अधिनियम}$)
        extracted_text = re.sub(r'\$\\text\{[^}]+\}\$\.?\n?', '', extracted_text)

        # 3. Removes nested square root text tags (e.g., $\sqrt{\text{१०}}.$)
        extracted_text = re.sub(r'\$\\sqrt\{\\text\{[^}]+\}\}\$\.?\n?', '', extracted_text)

        # 4. Removes ANY AI-generated metadata notes at the end of the page (Total Wipe)
        # Matches "Note: The document" and deletes everything until the end of the string.
        extracted_text = re.sub(r'Note:\s*The document.*$', '', extracted_text, flags=re.IGNORECASE | re.DOTALL)

        # 5. Clean up any lingering multiple blank lines caused by the removals
        extracted_text = re.sub(r'\n{3,}', '\n\n', extracted_text).strip()
        # ------------------------------------------------

        # Aggressive VRAM Flush
        del inputs
        del output_ids
        del generated_ids
        torch.cuda.empty_cache()
        gc.collect()

        return extracted_text

    def _parse_single_image(self, origin_image, save_dir, save_name, page_idx=0):
        # Run LightOn OCR directly on the raw image
        md_text = self._inference(origin_image)

        # Save Markdown
        md_file_path = os.path.join(save_dir, f"{save_name}.md")
        with open(md_file_path, "w", encoding="utf-8") as md_file:
            md_file.write(md_text)

        # Save the image for reference
        img_file_path = os.path.join(save_dir, f"{save_name}.jpg")
        origin_image.save(img_file_path)

        return {
            "page_no": page_idx,
            "md_content_path": md_file_path,
            "image_path": img_file_path
        }

    def parse_pdf(self, input_path, filename, save_dir, specific_pages=None):
        print(f"Loading PDF: {input_path}")
        doc = fitz.open(input_path)

        tasks = []
        for i in range(len(doc)):
            page_num = i + 1
            if specific_pages and page_num not in specific_pages:
                continue
            tasks.append((i, page_num))

        if not tasks:
            print("No pages matched the specific_pages filter. Exiting.")
            return []

        print(f"Parsing {len(tasks)} targeted pages using LightOn OCR sequentially...")

        results = []
        with tqdm(total=len(tasks), desc="Processing pages") as pbar:
            for i, page_num in tasks:
                page = doc.load_page(i)
                # Render at target DPI (Docs suggest ~2.77 scale which is roughly 200 DPI)
                pix = page.get_pixmap(dpi=self.dpi)
                pil_image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                save_name = f"{filename}_page_{page_num}"
                res = self._parse_single_image(pil_image, save_dir, save_name, page_idx=page_num)
                results.append(res)
                pbar.update(1)

        return results

    def parse_file(self, input_path, specific_pages=None):
        out_dir = os.path.abspath(self.output_dir)
        filename, file_ext = os.path.splitext(os.path.basename(input_path))
        save_dir = os.path.join(out_dir, filename)
        os.makedirs(save_dir, exist_ok=True)

        if file_ext.lower() == '.pdf':
            results = self.parse_pdf(input_path, filename, save_dir, specific_pages=specific_pages)
        elif file_ext.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            img = Image.open(input_path).convert("RGB")
            res = self._parse_single_image(img, save_dir, filename)
            results = [res]
        else:
            raise ValueError(f"File extension {file_ext} not supported.")

        print(f"\nParsing finished! Results saved to: {save_dir}")
        return results


def main():
    parser = argparse.ArgumentParser(description="Standalone LightOn OCR Parser")

    # Core Arguments
    parser.add_argument("input_path", type=str, help="Input PDF or image file path")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for PDF rendering")
    parser.add_argument("--pages_file", type=str, default=None, help="Text file with comma-separated pages")

    # DOTS.OCR COMPATIBILITY FLAGS (Ignored safely)
    parser.add_argument("--use_hf", type=str, default="true", help="Ignored")
    parser.add_argument("--num_thread", type=int, default=1, help="Ignored")

    args = parser.parse_args()

    # Early exit logic
    target_pages = None
    if args.pages_file:
        try:
            with open(args.pages_file, 'r') as f:
                content = f.read().strip()
                if content:
                    target_pages = [int(p.strip()) for p in content.split(',')]
                    print(f"Targeting specific pages: {target_pages}")
                else:
                    print(f"\n[+] The pages file '{args.pages_file}' is empty.")
                    print("[+] No pages require AI processing. Exiting immediately to save VRAM.\n")
                    sys.exit(0)
        except Exception as e:
            print(f"Warning: Could not read {args.pages_file}. Processing entire document. Error: {e}")

    # Initialize and run
    lighton_parser = LightOnOCRParser(output_dir=args.output, dpi=args.dpi)
    lighton_parser.parse_file(args.input_path, specific_pages=target_pages)


if __name__ == "__main__":
    main()