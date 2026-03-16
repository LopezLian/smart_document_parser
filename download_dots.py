from huggingface_hub import snapshot_download
import os

# Create the folder structure
os.makedirs("./weights", exist_ok=True)

print("⬇️  Starting download... This involves ~4GB of data.")
print("    Please wait, this depends on your internet speed...")

try:
    # UPDATED REPO ID: We are now using the official rednote-hilab repo
    model_path = snapshot_download(
        repo_id="rednote-hilab/dots.ocr",
        local_dir="./weights/DotsOCR",
        local_dir_use_symlinks=False
    )
    print(f"\n✅ Success! Model downloaded to: {model_path}")
    print("    You can now proceed to run the GPU script.")

except Exception as e:
    print(f"\n❌ Download failed: {e}")