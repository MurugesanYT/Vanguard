import os
from huggingface_hub import snapshot_download
from datasets import load_dataset

# Enable high-speed downloads if hf_transfer is installed
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def download_essentials():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. LLaVA Instruct
    print("\n--- Downloading LLaVA Instruct ---")
    try:
        snapshot_download(
            repo_id="liuhaotian/LLaVA-Instruct-150K",
            repo_type="dataset",
            local_dir=os.path.join(data_dir, "llava_instruct"),
            resume_download=True,
            max_workers=4
        )
        print("✅ LLaVA Instruct completed.")
    except Exception as e:
        print(f"❌ Error downloading LLaVA: {e}")

    # 2. ToolBench (using the community version suggested)
    print("\n--- Downloading ToolBench (tuandunghcmut/toolbench-v1) ---")
    try:
        # We use the datasets library to download and save it as it's more reliable for this specific repo
        dataset = load_dataset("tuandunghcmut/toolbench-v1", "default")
        dataset.save_to_disk(os.path.join(data_dir, "toolbench_v1"))
        print("✅ ToolBench completed.")
    except Exception as e:
        print(f"❌ Error downloading ToolBench: {e}")

    print("\n🚀 All downloads finished or resumed.")

if __name__ == "__main__":
    download_essentials()
