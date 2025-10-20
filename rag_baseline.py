import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate



# --- 組態設定 ---
# 讀取GOOGLE_API_KEY環境變數
# 如果尚未設定，在終端機中執行 (並重新啟動終端機): setx GOOGLE_API_KEY "your_key_here"
if not os.environ.get("GOOGLE_API_KEY"):
    raise RuntimeError("未在環境變數中找到 GOOGLE_API_KEY。請在執行前設定。")


# --- 模型與嵌入設定 ---
PDF_PATH = "要求/114-1 IR Final Project Requirements.pdf"   # 要讀取的檔案
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # 用來抽取特徵向量
LLM_MODEL = "gemini-2.5-flash" # 最後串接的LLM模型


# --- 文件分塊與向量儲存設定 ---
CHUNK_SIZE = 1000   # 每個chunk多少字
CHUNK_OVERLAP = 200 # 不同chunk間會有多少自重疊
VECTOR_STORE_PATH = "faiss_index"   # 向量儲存的地方
RETRIEVER_K = 4     # 檢索時要回傳的區塊數量

def setup_rag_pipeline():
    """
    完整的RAG流程，從讀取PDF到建立問答鏈
    如果找到已存在的向量儲存庫，將會直接載入以節省時間
    """
    if os.path.exists(VECTOR_STORE_PATH):
        print(f"載入已存在的向量儲存庫於: {VECTOR_STORE_PATH}")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vector_store = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"未找到向量儲存庫，將從頭建立一個新的: {PDF_PATH}")
        # 1. 載入並切割文件
        print("載入 PDF 中...")
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        
        print("將文件切割成多個區塊...")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        texts = text_splitter.split_documents(documents)
        
        # 2. 建立嵌入向量
        print(f"使用 {EMBEDDING_MODEL} 建立嵌入向量中...")
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        
        # 3. 建立並儲存向量儲存庫
        print("建立並儲存向量儲存庫 (FAISS)...")
        vector_store = FAISS.from_documents(texts, embeddings)
        vector_store.save_local(VECTOR_STORE_PATH)
        print(f"向量儲存庫已存於: {VECTOR_STORE_PATH}")

    # 4. 建立問答鏈
    print("使用最新的推薦方法建立問答鏈中...")
    # 根據儲存的Chunk的Embedding建立Retriever
    retriever = vector_store.as_retriever(search_kwargs={'k': RETRIEVER_K})

    # 初始化LLM
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)

    # 建立提示模板
    template = """僅根據以下內容來回答問題:{context}\n問題: {input}"""
    # 建立聊天模板
    prompt = ChatPromptTemplate.from_template(template)

    # 建立合併文件的鏈，檢索的時候，負責將檢索到的內容合併成一個大的文字塊，將該文字塊以及問題填入template後，向LLM提問、接收回答
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    # 建立最終的檢索鏈，這條鏈可以根據使用者輸入的問題在retriver中檢索相關的chunk，然後將檢索到的Chunk丟給combine_docs_chain進行提問
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    print("--- RAG 流程設定完成 ---")
    return qa_chain

def ask_question(qa_chain, query):
    """
    使用問答鏈來提問並印出答案。
    """
    print(f"\n問題: {query}")

    try:
        # 利用qa_chain完成相關chunk的檢索以及向LLM提問、接收答案的流程
        response = qa_chain.invoke({"input": query})
        print(f"\n答案: {response['answer']}")
        print("\n--- 參考來源 ---")
        for source in response["context"]:
            # 清理來源內容中的換行符，使其更易於閱讀
            cleaned_content = source.page_content.replace('\n', ' ').strip()
            source_content = cleaned_content[:200] + "..."
            print(f"頁碼: {source.metadata.get('page', 'N/A')}, 內容: \"{source_content}\"")

    except Exception as e:
        print(f"\n發生錯誤: {e}")
        print("請確認您的Google API金鑰是否正確且有效。")


if __name__ == "__main__":
    # --- 主要執行區 ---
    rag_pipeline = setup_rag_pipeline()
    
    # 範例問題
    example_query = "What is the submission deadline for the final project report?"
    ask_question(rag_pipeline, example_query)

    # 互動式問答迴圈
    print("\n已進入互動模式，輸入 'exit' 來離開。")
    while True:
        user_query = input("\n您的問題: ")
        if user_query.lower() == 'exit':
            break
        ask_question(rag_pipeline, user_query)