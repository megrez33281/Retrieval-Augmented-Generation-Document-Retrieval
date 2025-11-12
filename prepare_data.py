# 用於切割Chunk
import os
import json
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

# --- 組態設定 ---
DATASET_DIR = "Dataset"
OUTPUT_FILE = "chunks.json"
# 注意：SemanticChunker 不需要 chunk_size 和 overlap
# 它使用嵌入模型來決定切割點
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def create_chunks_file():
    """
    讀取資料夾中的所有PDF，將它們以語意方式切割成帶有ID的區塊，並儲存為JSON檔案。
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
        try:
            loader = PyPDFLoader(pdf_path)
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"讀取 {pdf_path} 失敗: {e}")

    if not all_docs:
        print("未能成功載入任何文件。")
        return
    
    print("\n使用語意切割器 (Semantic Chunker) 切割文件...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    text_splitter = SemanticChunker(embeddings)
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
        
    print(f"\n成功產生 {len(chunks_with_ids)} 個語意區塊，已儲存至 {OUTPUT_FILE}")

if __name__ == "__main__":
    create_chunks_file()
