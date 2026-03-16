import os
import ollama


class AILegalProofreader:
    def __init__(self):
        print("Connecting to local Llama 3.2 AI on your GPU...")
        try:
            # Verify the model is downloaded and ready
            ollama.show('llama3.2')
            print("Llama 3.2 is locked and loaded!")
        except Exception:
            print("WARNING: Model not found. Run 'ollama run llama3.2' in terminal first.")

    def proofread_block(self, text_block):
        if not text_block.strip():
            return text_block

        # This strict prompt stops the AI from hallucinating or chatting
        prompt = f"""You are a strict text correction engine for Indian Legal documents.
Fix the OCR spelling errors in the following text.

CRITICAL RULES:
1. DO NOT change names of people, judges, or places (e.g., Anuja, Prabhudessai, Pune).
2. DO NOT change Indian legal terms (e.g., putnidar, vakalatnama, suo moto).
3. DO NOT rephrase the sentence. Only fix broken spelling.
4. DO NOT add any conversational text. Output ONLY the corrected text.

Original Text:
{text_block}"""

        try:
            response = ollama.chat(model='llama3.2', messages=[
                {'role': 'user', 'content': prompt}
            ])
            return response['message']['content'].strip()
        except Exception as e:
            print(f"Error: {e}")
            return text_block

    def process_document(self, text):
        # Splitting by double-newline processes whole paragraphs for better context
        blocks = text.split('\n\n')
        corrected_blocks = []

        print(f"Proofreading {len(blocks)} blocks with Llama 3.2...")

        for i, block in enumerate(blocks):
            # Skip formatting markers
            if not block.strip() or block.startswith("--- PAGE") or block.startswith("[IMAGE") or block.startswith(
                    "[DIAGRAM"):
                corrected_blocks.append(block)
                continue

            print(f"  Fixing block {i + 1}/{len(blocks)}...")
            fixed_block = self.proofread_block(block)
            corrected_blocks.append(fixed_block)

        return '\n\n'.join(corrected_blocks)


# --- EXECUTION SCRIPT ---
if __name__ == "__main__":
    proofreader = AILegalProofreader()

    input_file = "extracted_output.txt"
    output_file = "final_perfected_output.txt"

    if os.path.exists(input_file):
        print(f"\nReading {input_file}...")
        with open(input_file, "r", encoding="utf-8") as f:
            raw_text = f.read()

        print("Applying Llama 3.2 corrections...")
        polished_text = proofreader.process_document(raw_text)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(polished_text)

        print(f"\nSuccess! Perfected text saved to {output_file}")
    else:
        print(f"Error: Could not find '{input_file}'.")