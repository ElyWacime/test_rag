import time
import sys

def step1_imports():
    print("1. Testing imports...", end=" ")
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.llms import Ollama
        print("✅")
        return True
    except Exception as e:
        print(f"❌ {e}")
        return False

def step2_ollama_connection():
    print("2. Testing Ollama connection...", end=" ")
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version", timeout=10)
        print(f"✅ (Version: {response.json().get('version', 'unknown')})")
        return True
    except Exception as e:
        print(f"❌ {e}")
        return False

def step3_bge_m3():
    print("3. Loading BGE-M3...", end=" ")
    sys.stdout.flush()
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        start = time.time()
        embeddings = OllamaEmbeddings(model="bge-m3")
        # Don't test embedding generation yet, just loading
        load_time = time.time() - start
        print(f"✅ ({load_time:.1f}s)")
        return True
    except Exception as e:
        print(f"❌ {e}")
        return False

def step4_llama():
    print("4. Loading Llama 3.2 3B...", end=" ")
    sys.stdout.flush()
    try:
        from langchain_community.llms import Ollama
        start = time.time()
        llm = Ollama(model="llama3.2:3b")
        load_time = time.time() - start
        print(f"✅ ({load_time:.1f}s)")
        return True
    except Exception as e:
        print(f"❌ {e}")
        return False

def step5_qwen():
    print("5. Loading Qwen Reranker...", end=" ")
    sys.stdout.flush()
    try:
        from langchain_community.llms import Ollama
        start = time.time()
        reranker = Ollama(model="dengcao/Qwen3-Reranker-0.6B:Q8_0")
        load_time = time.time() - start
        print(f"✅ ({load_time:.1f}s)")
        return True
    except Exception as e:
        print(f"❌ {e}")
        return False

def main():
    print("🔍 Minimal Step-by-Step Test")
    print("=" * 40)
    
    steps = [
        step1_imports,
        step2_ollama_connection,
        step3_bge_m3,
        step4_llama,
        step5_qwen
    ]
    
    for i, step in enumerate(steps, 1):
        print(f"Step {i}/5: ", end="")
        if not step():
            print(f"\n❌ Failed at step {i}")
            break
        time.sleep(1)  # Small delay between steps
    
    print("\n💡 If it hangs at a specific step, that model is likely the issue.")

if __name__ == "__main__":
    main()
