import os
os.environ["HF_HOME"] = "F:/hf"
import os
import subprocess
import sys
from huggingface_hub import snapshot_download, scan_cache_dir
from pathlib import Path

def locate_cached_hf_model():
    """Scans the local HF cache to find the path of the already downloaded model."""
    print(f"🔍 Searching for {HF_MODEL_ID} in local Hugging Face Hub cache...")
    
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == HF_MODEL_ID:
                # Find the 'main' revision or take the first available one
                revision = next((r for r in repo.revisions if r.snapshot_path), None)
                if not revision and repo.revisions:
                    revision = list(repo.revisions)[0]
                
                if revision and revision.snapshot_path:
                    local_path = Path(revision.snapshot_path)
                    print(f"✅ Found cached model locally at: {local_path}")
                    return local_path
    except Exception as e:
        print(f"⚠️ Error scanning cache: {e}")

    print(f"📥 Model not found in cache. Downloading directly into HF Hub directory...")
    downloaded_path = snapshot_download(
        repo_id=HF_MODEL_ID,
        revision="main"
    )
    final_path = Path(downloaded_path)
    print(f"✅ Model successfully cached in Hub structure at: {final_path}")

def setup_llama_cpp():
    """Clones llama.cpp repository and installs internal scripts dependencies."""
    if not os.path.exists(LLAMA_CPP_DIR):
        print("🐙 Cloning llama.cpp repository for conversion tools...")
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp.git", LLAMA_CPP_DIR], check=True)
    
    core_packages = ["numpy", "gguf", "transformers", "torch", "sentencepiece"]
    
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "--upgrade", "pip", "setuptools", "wheel"
    ], check=True)
    
    subprocess.run([
        sys.executable, "-m", "pip", "install", 
        "--only-binary=:all:", *core_packages
    ], check=True)

def convert_to_base_gguf(model_path):
    if not os.path.exists(INTERMEDIATE_F16_GGUF):
        print("⚡ Converting HF model weights into base F16 GGUF...")
        convert_script = os.path.join(LLAMA_CPP_DIR, "convert_hf_to_gguf.py")
        
        command = [
            sys.executable, convert_script,
            str(model_path),
            "--outfile", str(INTERMEDIATE_F16_GGUF),
            "--outtype", "f16"  # Must be f16/f32 baseline for the quantizer to ingest 
        ]
        
        subprocess.run(command, check=True)
        print(f"✅ Intermediate F16 baseline generated: {INTERMEDIATE_F16_GGUF}")
    else:
        print("Intermediate F16 baseline is already generated.")

def compile_binary_from_source():
    build_dir = LLAMA_CPP_DIR / "build"
    if not os.path.exists(build_dir):
        print("🛠️ Invoking CMake configuration parser...")
        
        # Generate the compiler makefiles / build solutions
        subprocess.run([
            "cmake", 
            "-B", str(build_dir), 
            "-S", str(LLAMA_CPP_DIR)
        ], check=True)
        
        print("🏗️ Compiling llama-quantize executable natively via source files...")
        # Compile the binaries using native multi-core compilation configuration
        subprocess.run([
            "cmake", 
            "--build", str(build_dir), 
            "--config", "Release", 
            "-j"
        ], check=True)
    else:
        print("Binary files are already created from source code")


def quantize_gguf():
    if not os.path.exists(OUTPUT_GGUF_NAME):
        print(f"📉 Compressing baseline file down to {QUANTIZATION_TYPE}...")
        quantize_exe = LLAMA_CPP_DIR / "build" / "bin" / "Release" / "llama-quantize"
            
        command = [
            str(quantize_exe),
            str(INTERMEDIATE_F16_GGUF),
            str(OUTPUT_GGUF_NAME),
            QUANTIZATION_TYPE
        ]
        
        subprocess.run(command, check=True)
        print(f"🎉 Success! Final GGUF model created: {OUTPUT_GGUF_NAME}")   
    else:
        print("GGUF file was already created.")

if __name__ == "__main__":
    HF_MODEL_ID = "Algorithmica/smollm2-thinking"

    # The final quantization type you want
    QUANTIZATION_TYPE = "q4_k_m"

    BASE_DIR = Path("F:/gguf")
    LLAMA_CPP_DIR = BASE_DIR / "llama_cpp_repo"

    # Step 1 Target (Temporary baseline unquantized file)
    INTERMEDIATE_F16_GGUF = BASE_DIR / "model_f16_base.gguf"
    # Step 2 Target (Final desired output file)
    OUTPUT_GGUF_NAME = BASE_DIR / f"model_{QUANTIZATION_TYPE}.gguf"

    try:
        # Resolve target files 
        local_model_path = locate_cached_hf_model()
        print(local_model_path)
        
        # Pull python dependencies & compile binary toolchain
        setup_llama_cpp()

        # Compile binary from source
        compile_binary_from_source()
        
        # HF -> F16 GGUF
        convert_to_base_gguf(local_model_path)
        
        # F16 GGUF -> Q4_K_M GGUF
        quantize_gguf()
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Script failed during CLI command execution: {e}")