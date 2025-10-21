# 用於切割Chunk
import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 組態設定 ---
DATASET_DIR = "Dataset"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
OUTPUT_FILE = "chunks.json"

def create_chunks_file():
    """
    讀取資料夾中的所有PDF，將它們切割成帶有ID的區塊，並儲存為JSON檔案。
    """
    pdf_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.pdf')]
    if not pdf_files:
        print(f"在 {DATASET_DIR} 中找不到任何 PDF 檔案。")
        return

    all_docs = []
    print("載入 PDF 檔案中...")
    for pdf_file in pdf_files:
        pdf_path = os.path.join(DATASET_DIR, pdf_file)
        print(f" - 正在載入: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        all_docs.extend(loader.load())
    
    print("\n切割文件...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    texts = text_splitter.split_documents(all_docs)
    
    # 為每個區塊加上唯一的 ID
    chunks_with_ids = []
    for i, doc in enumerate(texts):
        # 清理一下 metadata 的 source 路徑
        doc.metadata["source"] = os.path.basename(doc.metadata["source"])
        
        chunks_with_ids.append({
            "chunk_id": i,
            "content": doc.page_content,
            "metadata": doc.metadata
        })
    
    # 寫入 JSON 檔案
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(chunks_with_ids, f, ensure_ascii=False, indent=4)
        
    print(f"\n成功產生 {len(chunks_with_ids)} 個區塊，已儲存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    create_chunks_file()
