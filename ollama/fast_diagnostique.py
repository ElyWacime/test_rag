import requests
import time
import threading
import sys

def test_with_timeout(func, timeout=30, *args, **kwargs):
    """Run a function with timeout"""
    result = [None]
    exception = [None]
    
    def worker():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e
    
    thread = threading.Thread(target=worker)
    thread.daemon = True
    thread.start()
    thread.join(timeout)
    
    if thread.is_alive():
        return None, f"TIMEOUT after {timeout} seconds"
    elif exception[0] is not None:
        return None, str(exception[0])
    else:
        return result[0], None

def quick_ollama_check():
    """Quick check if Ollama is responsive"""
    print("🔌 Quick Ollama check...")
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model['name'] for model in models]
            print(f"✅ Ollama running with {len(models)} models")
            return True, model_names
        else:
            return False, f"HTTP {response.status_code}"
    except Exception as e:
        return False, str(e)

def test_model_loading():
    """Test if models can be loaded without hanging"""
    print("\n📥 Testing model loading...")
    
    # Test 1: BGE-M3 embeddings
    print("   Testing BGE-M3...", end=" ", flush=True)
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        start_time = time.time()
        embeddings = OllamaEmbeddings(model="bge-m3")
        # Quick embedding test
        test_embedding = embeddings.embed_query("test")
        load_time = time.time() - start_time
        print(f"✅ ({load_time:.1f}s)")
    except Exception as e:
        print(f"❌ ({str(e)})")
        return False
    
    # Test 2: Llama 3.2 3B
    print("   Testing Llama 3.2 3B...", end=" ", flush=True)
    try:
        from langchain_community.llms import Ollama
        start_time = time.time()
        llm = Ollama(model="llama3.2:3b", num_predict=10)  # Very short output
        # Quick generation test
        response = llm.invoke("Say 'hi'")
        load_time = time.time() - start_time
        print(f"✅ ({load_time:.1f}s)")
    except Exception as e:
        print(f"❌ ({str(e)})")
        return False
    
    # Test 3: Qwen Reranker
    print("   Testing Qwen Reranker...", end=" ", flush=True)
    try:
        from langchain_community.llms import Ollama
        start_time = time.time()
        reranker = Ollama(model="dengcao/Qwen3-Reranker-0.6B:Q8_0", num_predict=5)
        # Quick test
        response = reranker.invoke("test")
        load_time = time.time() - start_time
        print(f"✅ ({load_time:.1f}s)")
    except Exception as e:
        print(f"❌ ({str(e)})")
        return False
    
    return True

def check_system_resources():
    """Check if system has enough resources"""
    print("\n💻 System resource check...")
    try:
        import psutil
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        print(f"   RAM: {memory.available / (1024**3):.1f}GB available")
        print(f"   CPU: {cpu_percent}% usage")
        
        if memory.available < 2 * 1024**3:  # Less than 2GB available
            print("   ⚠️  Low RAM - models might struggle to load")
            return False
        return True
    except ImportError:
        print("   ℹ️  Install psutil for detailed resource monitoring: pip install psutil")
        return True

def main():
    print("🚀 Fast Diagnostic Test")
    print("=" * 50)
    
    # Check Ollama first
    ollama_ok, ollama_msg = quick_ollama_check()
    if not ollama_ok:
        print(f"❌ Ollama issue: {ollama_msg}")
        print("💡 Make sure Ollama is running: ollama serve")
        return
    
    # Check system resources
    check_system_resources()
    
    # Test model loading with timeouts
    print("\n⏳ Testing model loading (this may take a few minutes for first load)...")
    
    models_loaded = test_model_loading()
    
    if models_loaded:
        print("\n🎉 All models loaded successfully!")
        print("\n💡 If your RAG system is still hanging, it might be:")
        print("   - Processing a large number of emails")
        print("   - Building the FAISS index for the first time")
        print("   - The models are still warming up")
    else:
        print("\n❌ Some models failed to load")
        print("\n🔧 Troubleshooting steps:")
        print("   1. Restart Ollama: ollama serve")
        print("   2. Check if models are downloaded: ollama list")
        print("   3. Try pulling models again: ollama pull <model-name>")
        print("   4. Check available RAM - close other applications")

if __name__ == "__main__":
    main()
