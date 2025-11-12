# 讀取切割的Chunk，回答問題
import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# --- 組態設定 ---
# 讀取GOOGLE_API_KEY環境變數
# 如果尚未設定，在終端機中執行 (並重新啟動終端機): setx GOOGLE_API_KEY "your_key_here"
if not os.environ.get("GOOGLE_API_KEY"):
    raise RuntimeError("未在環境變數中找到 GOOGLE_API_KEY。請在執行前設定。")


# --- 模型與嵌入設定 ---
PDF_PATH = "要求/114-1 IR Final Project Requirements.pdf"   # 要讀取的檔案
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2" # 用來抽取特徵向量
LLM_MODEL = "gemini-2.5-flash" # 最後串接的LLM模型


# --- 文件分塊與向量儲存設定 ---
CHUNK_SIZE = 1000   # 每個chunk多少字
CHUNK_OVERLAP = 200 # 不同chunk間會有多少自重疊
VECTOR_STORE_PATH = "faiss_index"   # 向量儲存的地方
RETRIEVER_K = 3     # 檢索時要回傳的區塊數量


# --- 全局讀取一次 chunks.json ---
CHUNKS_FILE = 'chunks.json'
if os.path.exists(CHUNKS_FILE):
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        ALL_CHUNKS = json.load(f)
else:
    ALL_CHUNKS = None
    print(f"警告: {CHUNKS_FILE} 不存在。請先執行 prepare_data.py 來產生 chunk 資料。")


def setup_rag_pipeline(source_file_name, embedding_model):
    """
    為指定的來源檔案和嵌入模型建立 RAG 流程。
    """
    if ALL_CHUNKS is None:
        raise RuntimeError(f"{CHUNKS_FILE} 未載入，無法繼續")

    # 根據來源檔案和模型名稱動態產生向量儲存庫路徑
    file_name_base = os.path.splitext(source_file_name)[0]
    # 將模型名稱中的斜線替換成底線，以建立有效的資料夾名稱
    sanitized_model_name = embedding_model.replace('/', '_')
    vector_store_path = f"faiss_index_{file_name_base}_{sanitized_model_name}"

    if os.path.exists(vector_store_path):
        print(f"載入已存在的向量儲存庫於: {vector_store_path}")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"為 {source_file_name} 使用 {embedding_model} 建立新的向量儲存庫")
        
        # 1. 從 ALL_CHUNKS 中篩選出屬於此來源的 chunks
        print(f"從 {CHUNKS_FILE} 篩選來源為 '{source_file_name}' 的區塊")
        source_chunks_json = [chunk for chunk in ALL_CHUNKS if chunk['metadata']['source'] == source_file_name]
        
        if not source_chunks_json:
            raise ValueError(f"在 {CHUNKS_FILE} 中找不到任何來源為 '{source_file_name}' 的區塊")

        # 將 JSON 物件轉換回 LangChain 的 Document 物件，並確保 chunk_id 被包含在 metadata 中
        source_documents = []
        for chunk in source_chunks_json:
            new_metadata = chunk['metadata'].copy()
            new_metadata['chunk_id'] = chunk['chunk_id']
            source_documents.append(Document(page_content=chunk['content'], metadata=new_metadata))

        # 2. 建立嵌入向量
        print(f"使用 {embedding_model} 建立嵌入向量中")
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # 3. 建立並儲存向量儲存庫
        print("建立並儲存向量儲存庫 (FAISS)")
        vector_store = FAISS.from_documents(source_documents, embeddings)
        vector_store.save_local(vector_store_path)
        print(f"向量儲存庫已存於: {vector_store_path}")

    # 4. 建立問答鏈
    # 根據儲存的Chunk的Embedding建立Retriever
    retriever = vector_store.as_retriever(search_kwargs={'k': RETRIEVER_K})

    # 初始化LLM
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)

    # 建立提示模板
    template = '''僅根據以下內容來回答問題:{context}\n問題: {input}'''
    # 建立聊天模板
    prompt = ChatPromptTemplate.from_template(template)

    # 建立合併文件的鏈，檢索的時候，負責將檢索到的內容合併成一個大的文字塊，將該文字塊以及問題填入template後，向LLM提問、接收回答
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    # 建立最終的檢索鏈，這條鏈可以根據使用者輸入的問題在retriver中檢索相關的chunk，然後將檢索到的Chunk丟給combine_docs_chain進行提問
    qa_chain = create_retrieval_chain(retriever, combine_docs_chain)
    
    print(f"--- 為 {source_file_name} 的 RAG 流程設定完成 ---")
    return {"qa_chain": qa_chain, "retriever": retriever}


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
        print("請確認Google API金鑰是否正確且有效")


if __name__ == "__main__":
    # --- 主要執行區 ---
    if ALL_CHUNKS is None:
        exit()
        
    # 注意：全域變數 PDF_PATH 在此互動模式下已無直接作用
    # 這裡我們示範如何為指定的檔案建立互動問答
    pdf_to_query_name = "2025 Generative Information Retrieval HW1.pdf"
    print(f"--- 正在為 {pdf_to_query_name} 建立互動式問答環境 ---")

    pipeline_components = setup_rag_pipeline(pdf_to_query_name, EMBEDDING_MODEL)
    rag_pipeline = pipeline_components["qa_chain"]
    
    # 範例問題
    example_query = "What is the submission deadline for the final project report?"
    ask_question(rag_pipeline, example_query)

    # 互動式問答迴圈
    print("\n已進入互動模式，輸入 'exit' 來離開")
    while True:
        user_query = input("\n輸入問題: ")
        if user_query.lower() == 'exit':
            break
        ask_question(rag_pipeline, user_query)
