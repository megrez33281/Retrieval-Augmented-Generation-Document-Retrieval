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
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 組態設定 ---
# 讀取GOOGLE_API_KEY環境變數
# 如果尚未設定，在終端機中執行 (並重新啟動終端機): setx GOOGLE_API_KEY "your_key_here"
if not os.environ.get("GOOGLE_API_KEY"):
    raise RuntimeError("未在環境變數中找到 GOOGLE_API_KEY。請在執行前設定。")


# --- 模型與嵌入設定 ---
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # 用來抽取特徵向量
LLM_MODEL = "gemini-2.5-flash" # 最後串接的LLM模型


# --- 文件分塊與向量儲存設定 ---
CHUNK_SIZE = 1000   # 每個chunk多少字
CHUNK_OVERLAP = 200 # 不同chunk間會有多少自重疊
KNOWLEDGE_DIR = "Knowledge" # 補充知識庫的資料夾
RETRIEVER_K = 3     # 檢索時要回傳的區塊數量


# --- 全局讀取一次 chunks.json ---
CHUNKS_FILE = 'chunks.json'
if os.path.exists(CHUNKS_FILE):
    with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
        ALL_CHUNKS = json.load(f)
else:
    ALL_CHUNKS = None
    print(f"警告: {CHUNKS_FILE} 不存在。請先執行 prepare_data.py 來產生 chunk 資料。")


def setup_primary_retriever(source_file_name, embedding_model_name, embeddings):
    """
    為指定的主要來源檔案建立 RAG 流程中的檢索器。
    """
    if ALL_CHUNKS is None:
        raise RuntimeError(f"{CHUNKS_FILE} 未載入，無法繼續")

    # 根據來源檔案和模型名稱動態產生向量儲存庫路徑
    file_name_base = os.path.splitext(source_file_name)[0]
    sanitized_model_name = embedding_model_name.replace('/', '_')
    vector_store_path = f"faiss_index_{file_name_base}_{sanitized_model_name}"

    if os.path.exists(vector_store_path):
        print(f"載入已存在的主要知識庫於: {vector_store_path}")
        vector_store = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
    else:
        print(f"為 {source_file_name} 使用 {embedding_model_name} 建立新的主要知識庫")
        
        source_chunks_json = [c for c in ALL_CHUNKS if c['metadata']['source'] == source_file_name]
        if not source_chunks_json:
            raise ValueError(f"在 {CHUNKS_FILE} 中找不到任何來源為 '{source_file_name}' 的區塊")

        source_documents = [
            Document(page_content=c['content'], metadata={**c['metadata'], 'chunk_id': c['chunk_id']})
            for c in source_chunks_json
        ]

        print(f"正在為主要知識庫建立嵌入向量...")
        vector_store = FAISS.from_documents(source_documents, embeddings)
        vector_store.save_local(vector_store_path)
        print(f"主要知識庫已存於: {vector_store_path}")

    return vector_store.as_retriever(search_kwargs={'k': RETRIEVER_K})


from langchain_experimental.text_splitter import SemanticChunker


def setup_knowledge_retriever(knowledge_dir, embedding_model_name, embeddings):
    """
    為補充知識庫資料夾中的所有 PDF 建立一個記憶體內的檢索器。
    """
    print("\n--- 正在設定補充知識庫 ---")
    pdf_files = [os.path.join(knowledge_dir, f) for f in os.listdir(knowledge_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("在 Knowledge 資料夾中找不到任何 PDF 檔案，將跳過補充知識庫。")
        return None

    all_docs = []
    print("載入補充知識 PDF 檔案中...")
    for pdf_path in pdf_files:
        try:
            print(f" - 正在載入: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"讀取 {pdf_path} 失敗: {e}")
    
    if not all_docs:
        print("無法載入任何補充文件內容。")
        return None

    print("使用語意切割器 (Semantic Chunker) 切割補充知識文件...")
    text_splitter = SemanticChunker(embeddings)
    texts = text_splitter.split_documents(all_docs)
    
    print(f"正在為 {len(texts)} 個補充區塊建立記憶體內向量儲存庫 (FAISS)...")
    knowledge_vector_store = FAISS.from_documents(texts, embeddings)
    print("補充知識庫設定完成。")
    
    return knowledge_vector_store.as_retriever(search_kwargs={'k': RETRIEVER_K})


# --- 提示工程 ---
ADVANCED_PROMPT_TEMPLATE = """
請根據以下提供的兩種知識來回答問題。

你的任務是：
1.  首先，根據「主要知識」找出最直接、最核心的答案。
2.  接著，檢視「補充知識」。如果它提供了與問題高度相關的額外細節、定義或背景，請用它來豐富你的答案，讓回答更完整。
3.  如果「補充知識」與問題無關，或只是重複相同的資訊，則完全忽略它。
4.  你的回答應簡潔且準確，先給出直接答案，再進行必要的補充。
5.  **重要提示：如果你認為根據提供的知識無法充分回答問題，請不要嘗試編造答案。請輸出以下特殊標記，並在標記後提供一個你認為可以幫助檢索到更好資訊的「優化後的問題」。優化後的問題請使用英文。**
    **格式範例：[QUERY_REWRITE] Optimized English Question**

[主要知識]:
{primary_context}

[補充知識]:
{supplemental_context}

問題: {input}
答案:
"""

# --- 查詢重寫設定 ---
QUERY_REWRITE_TAG = "[QUERY_REWRITE]"
MAX_RETRIES = 2 # 最多重試一次

# --- QA 日誌設定 ---
QA_LOG_FILE = "qa_log.json"

def load_qa_log():
    """載入 QA 日誌檔案。"""
    if os.path.exists(QA_LOG_FILE):
        try:
            with open(QA_LOG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            print(f"警告: {QA_LOG_FILE} 檔案損壞或格式不正確，將重新建立。")
            return []
    return []

def save_qa_log(log_data):
    """儲存 QA 日誌檔案。"""
    with open(QA_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=4)

def collect_sources_info(docs):
    """從 LangChain Document 物件中收集來源資訊。"""
    sources_info = []
    for doc in docs:
        sources_info.append({
            "source": os.path.basename(doc.metadata.get('source', 'N/A')),
            "page": doc.metadata.get('page', 'N/A')
        })
    return sources_info

def ask_question(query, primary_retriever, knowledge_retriever, llm, qa_log, retry_count=0):
    """
    使用主要和補充檢索器來提問，並使用進階提示模板生成答案。
    如果 LLM 建議重寫問題，則會進行重試。
    """
    print(f"\n問題: {query}")

    try:
        # 1. 分別從兩個知識庫檢索
        print("正在從主要知識庫檢索...")
        primary_docs = primary_retriever.invoke(query)
        
        knowledge_docs = []
        if knowledge_retriever:
            print("正在從補充知識庫檢索...")
            knowledge_docs = knowledge_retriever.invoke(query)
        
        # 2. 格式化文件內容以用於提示
        primary_context_str = "\n\n".join([f"來源: {os.path.basename(doc.metadata.get('source', 'N/A'))}, 頁碼: {doc.metadata.get('page', 'N/A')}\n內容: {doc.page_content}" for doc in primary_docs])
        supplemental_context_str = "\n\n".join([f"來源: {os.path.basename(doc.metadata.get('source', 'N/A'))}, 頁碼: {doc.metadata.get('page', 'N/A')}\n內容: {doc.page_content}" for doc in knowledge_docs])

        if not primary_context_str and not supplemental_context_str:
            final_answer = "抱歉，在所有知識庫中都找不到相關資訊。"
            print(f"\n答案: {final_answer}")
            # 記錄這次的問答
            qa_log.append({
                "question": query,
                "answer": final_answer,
                "sources": []
            })
            save_qa_log(qa_log)
            return

        # 3. 建立並執行鏈
        prompt = ChatPromptTemplate.from_template(ADVANCED_PROMPT_TEMPLATE)
        chain = prompt | llm
        
        print("已組合上下文，正在使用進階提示生成答案...")
        response = chain.invoke({
            "primary_context": primary_context_str if primary_docs else "無",
            "supplemental_context": supplemental_context_str if knowledge_docs else "無",
            "input": query
        })
        
        # AIMessage 物件的回應在 content 屬性中
        answer = response.content
        
        # 檢查是否需要重寫問題
        if QUERY_REWRITE_TAG in answer and retry_count < MAX_RETRIES:
            print(f"\n偵測到查詢重寫請求 ({QUERY_REWRITE_TAG})。")
            rewritten_query = answer.split(QUERY_REWRITE_TAG, 1)[1].strip()
            print(f"優化後的問題: {rewritten_query}")
            print(f"正在使用優化後的問題重新檢索 (重試次數: {retry_count + 1}/{MAX_RETRIES})...")
            # 遞迴呼叫 ask_question 進行重試
            ask_question(rewritten_query, primary_retriever, knowledge_retriever, llm, qa_log, retry_count + 1)
        elif QUERY_REWRITE_TAG in answer and retry_count >= MAX_RETRIES:
            print(f"\n已達到最大重試次數 ({MAX_RETRIES})。將顯示原始 LLM 回應。")
            print(f"\n答案: {answer}")
            # 記錄這次的問答
            all_sources_for_log = collect_sources_info(primary_docs + knowledge_docs)
            qa_log.append({
                "question": query,
                "answer": answer,
                "sources": all_sources_for_log
            })
            save_qa_log(qa_log)
        else:
            print(f"\n答案: {answer}")
            
            # 顯示參考來源
            print("\n--- 參考來源 ---")
            all_sources_for_log = []
            if primary_docs:
                print("  --- 主要知識來源 ---")
                for doc in primary_docs:
                    source_info = {"source": os.path.basename(doc.metadata.get('source', 'N/A')), "page": doc.metadata.get('page', 'N/A')}
                    print(f"    - {source_info['source']} (頁碼: {source_info['page']})")
                    all_sources_for_log.append(source_info)
            if knowledge_docs:
                print("  --- 補充知識來源 ---")
                for doc in knowledge_docs:
                    source_info = {"source": os.path.basename(doc.metadata.get('source', 'N/A')), "page": doc.metadata.get('page', 'N/A')}
                    print(f"    - {source_info['source']} (頁碼: {source_info['page']})")
                    all_sources_for_log.append(source_info)
            
            # 記錄這次的問答
            qa_log.append({
                "question": query,
                "answer": answer,
                "sources": all_sources_for_log
            })
            save_qa_log(qa_log)

    except Exception as e:
        print(f"\n處理問題時發生錯誤: {e}")
        print("請確認 Google API 金鑰是否正確且有效。")
        return
if __name__ == "__main__":
    if ALL_CHUNKS is None:
        exit()
        
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)

    # --- 1. 初始化共用元件 ---
    print("--- 正在初始化 RAG 系統 ---")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL)
    
    # --- 2. 設定主要和補充檢索器 ---
    pdf_to_query_name = "2025 Generative Information Retrieval HW1"
    print(f"\n--- 正在為 '{pdf_to_query_name}' 設定主要知識庫 ---")
    primary_retriever = setup_primary_retriever(pdf_to_query_name, EMBEDDING_MODEL, embeddings)
    
    knowledge_retriever = setup_knowledge_retriever(KNOWLEDGE_DIR, EMBEDDING_MODEL, embeddings)
    
    # --- 3. 載入 QA 日誌 ---
    qa_log = load_qa_log()

    print("\n--- RAG 系統已就緒 ---")

    # --- 4. 互動式問答迴圈 ---
    print("\n已進入互動模式，輸入 'exit' 來離開。")
    while True:
        user_query = input("\n輸入問題: ")
        if user_query.lower() == 'exit':
            break
        if not user_query.strip():
            continue
        
        ask_question(
            query=user_query,
            primary_retriever=primary_retriever,
            knowledge_retriever=knowledge_retriever,
            llm=llm,
            qa_log=qa_log
        )
