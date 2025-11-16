import requests
import json
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

def test_ollama_connection():
    """Test basic Ollama connection"""
    print("🔌 Testing Ollama connection...")
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            models = response.json().get("models", [])
            print("✅ Ollama is running!")
            print(f"📚 Found {len(models)} models:")
            for model in models:
                print(f"   - {model['name']}")
            return True
        else:
            print("❌ Ollama responded with error")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("💡 Make sure Ollama is running: ollama serve")
        return False

def test_bge_m3_embeddings():
    """Test BGE-M3 embeddings"""
    print("\n🧠 Testing BGE-M3 embeddings...")
    try:
        embeddings = OllamaEmbeddings(
            model="bge-m3",
            base_url="http://localhost:11434"
        )
        
        # Test with a simple sentence
        test_text = "This is a test sentence for embedding generation."
        embedding = embeddings.embed_query(test_text)
        
        print(f"✅ BGE-M3 embeddings working!")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        return True
        
    except Exception as e:
        print(f"❌ BGE-M3 embeddings failed: {e}")
        return False

def test_llama_3b_generation():
    """Test Llama 3.2 3B generation"""
    print("\n🤖 Testing Llama 3.2 3B generation...")
    try:
        llm = Ollama(
            model="llama3.2:3b",
            base_url="http://localhost:11434",
            temperature=0.1,
            num_predict=100
        )
        
        # Simple test prompt
        test_prompt = "Say 'Hello world' in a creative way."
        response = llm.invoke(test_prompt)
        
        print("✅ Llama 3.2 3B generation working!")
        print(f"   Response: {response.strip()}")
        return True
        
    except Exception as e:
        print(f"❌ Llama 3.2 3B generation failed: {e}")
        return False

def test_qwen_reranker():
    """Test Qwen3 reranker"""
    print("\n🎯 Testing Qwen3-Reranker...")
    try:
        reranker = Ollama(
            model="dengcao/Qwen3-Reranker-0.6B:Q8_0",
            base_url="http://localhost:11434",
            temperature=0
        )
        
        # Test reranking with a simple query and document
        test_query = "machine learning"
        test_document = "Artificial intelligence and machine learning are transforming the world."
        
        prompt = f"""Score the relevance between this query and document on a scale of 0.0 to 1.0.

Query: {test_query}

Document: {test_document}

Relevance score: """
        
        response = reranker.invoke(prompt)
        
        print("✅ Qwen3-Reranker working!")
        print(f"   Query: '{test_query}'")
        print(f"   Document: '{test_document}'")
        print(f"   Response: {response.strip()}")
        return True
        
    except Exception as e:
        print(f"❌ Qwen3-Reranker failed: {e}")
        return False

def test_email_rag_components():
    """Test components that will be used in the RAG system"""
    print("\n📧 Testing RAG components with email data...")
    try:
        # Test embeddings with email-like content
        embeddings = OllamaEmbeddings(model="bge-m3")
        llm = Ollama(model="llama3.2:3b")
        
        email_content = "Meeting scheduled for tomorrow at 2 PM about project discussion."
        embedding = embeddings.embed_query(email_content)
        
        # Test LLM with email context
        rag_prompt = f"""Based on this email content, when is the meeting?

Email: {email_content}

Answer:"""
        
        response = llm.invoke(rag_prompt)
        
        print("✅ RAG components working!")
        print(f"   Email content: '{email_content}'")
        print(f"   LLM response: {response.strip()}")
        return True
        
    except Exception as e:
        print(f"❌ RAG components test failed: {e}")
        return False

def main():
    print("🚀 Starting Model Tests...")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 5
    
    # Run all tests
    if test_ollama_connection():
        tests_passed += 1
    
    if test_bge_m3_embeddings():
        tests_passed += 1
        
    if test_llama_3b_generation():
        tests_passed += 1
        
    if test_qwen_reranker():
        tests_passed += 1
        
    if test_email_rag_components():
        tests_passed += 1
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your RAG system should work perfectly!")
    elif tests_passed >= 3:
        print("⚠️  Most tests passed. Some components might need attention.")
    else:
        print("❌ Multiple tests failed. Please check your Ollama setup.")
    
    print("\n💡 Next step: Run your RAG system with: python rag_system.py")

if __name__ == "__main__":
    main()
