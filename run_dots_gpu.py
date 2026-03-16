import sys
import os
import json
import time
import gc
import fitz  # PyMuPDF
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# --- CONFIGURATION ---
MODEL_PATH = "./weights/DotsOCR"
MAX_DIMENSION = 1536  # <--- CRITICAL: Increased to 1536. 1024 was too blurry.
MAX_TOKENS = 800  # Reduced slightly to save memory for the larger image
COOLDOWN = 5
OUTPUT_FILE = "final_output.json"

# --- 1. SETUP GPU ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"⚙️  Hardware: {DEVICE.upper()}")

print(f"Loading DotsOCR from {MODEL_PATH}...")
try:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("✅ Model loaded.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)


def flush_memory():
    """Aggressively cleans GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()


def resize_for_vram(image_path):
    with Image.open(image_path) as img:
        # Resize to 1536px (Riskier for VRAM, but necessary for accuracy)
        ratio = min(MAX_DIMENSION / img.width, MAX_DIMENSION / img.height)
        if ratio < 1:
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            img.save(image_path)


def run_dots_ocr(image_path):
    flush_memory()
    resize_for_vram(image_path)

    # --- STRICT NO-HALLUCINATION PROMPT ---
    prompt_text = """Analyze the text in this image.
    1. If the text is too blurry or unreadable, return an empty list: [].
    2. DO NOT GUESS. DO NOT MAKE UP TEXT.
    3. Only transcribe words you can clearly see.
    4. Output valid JSON with 'bbox', 'category', and 'text'."""

    messages = [
        {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": prompt_text}]}]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    flush_memory()

    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                do_sample=False  # Deterministic (Robot mode, no creativity)
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        del inputs
        del generated_ids
        flush_memory()

        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text

    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print("💥 GPU Out of Memory! The image is too big.")
            return "[]"  # Return empty on crash
        else:
            raise e


# --- MAIN ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parser.py <path_to_pdf>")
        sys.exit(1)

    input_path = sys.argv[1]

    # Force delete old file
    if os.path.exists(OUTPUT_FILE):
        try:
            os.remove(OUTPUT_FILE)
        except:
            pass

    doc = fitz.open(input_path)
    os.makedirs("temp_images", exist_ok=True)

    print(f"🚀 Starting HIGH-RES processing of {len(doc)} pages...")

    results = []

    for i in range(len(doc)):
        page_num = i + 1
        print(f"📄 Page {page_num}: Converting...", end=" ")

        page = doc.load_page(i)
        pix = page.get_pixmap(dpi=150)  # Higher DPI source
        img_path = f"temp_images/page_{page_num}.jpg"
        pix.save(img_path)

        try:
            start_t = time.time()
            raw_output = run_dots_ocr(img_path)
            clean_json = raw_output.replace("```json", "").replace("```", "").strip()

            # Simple validation to catch bad JSON
            if not clean_json.startswith("["):
                clean_json = "[]"

            results.append({"page": page_num, "raw_output": clean_json})
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)

            elapsed = time.time() - start_t
            print(f"✅ Done ({elapsed:.1f}s)")

            if i < len(doc) - 1:
                print(f"❄️  Cooling down...", end="\r")
                time.sleep(COOLDOWN)

        except Exception as e:
            print(f"❌ Failed: {e}")
            flush_memory()

    print(f"\n🏁 All Done! Saved to {OUTPUT_FILE}")