
import os
import json
from langchain_huggingface import HuggingFaceEmbeddings
from rag_baseline import (
    setup_primary_retriever, 
    setup_knowledge_retriever, 
    EMBEDDING_MODEL,
    KNOWLEDGE_DIR
)

def evaluate_retriever(golden_set, embedding_model):
    """
    使用黃金標準集和指定的嵌入模型來評估主要+補充知識庫的組合檢索性能。
    """
    recall_scores = []
    mrr_scores = []
    
    questions_by_source = {}
    for item in golden_set:
        if not item["relevant_chunk_ids"]:
            print(f"警告: 問題 \"{item['question'][:30]}...\" 的 relevant_chunk_ids 為空，將跳過此問題的評估。")
            continue
        
        source_filename = item["source_file"]
        if source_filename not in questions_by_source:
            questions_by_source[source_filename] = []
        questions_by_source[source_filename].append(item)

    if not questions_by_source:
        print("錯誤: 沒有可供評估的問題。請檢查 golden_set.json 是否已正確標記。")
        return 0.0, 0.0

    print(f"\n--- 使用模型: {embedding_model} ---")
    print(f"開始評估 {len(golden_set)} 個問題，來源分為 {len(questions_by_source)} 個檔案...")

    # --- 初始化共用元件 ---
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    # --- 設定補充知識庫 (只需一次) ---
    # 確保 Knowledge 資料夾存在
    os.makedirs(KNOWLEDGE_DIR, exist_ok=True)
    knowledge_retriever = setup_knowledge_retriever(KNOWLEDGE_DIR, embedding_model, embeddings)

    # --- 迭代每個主要知識庫來源 ---
    for source_filename, questions in questions_by_source.items():
        print(f"\n--- 正在處理主要來源: {source_filename} ---")
        
        # 為這個主要來源檔案建立專屬的 retriever
        primary_retriever = setup_primary_retriever(source_filename, embedding_model, embeddings)

        # 在這個 retriever 組合上評估所有相關問題
        for item in questions:
            question = item["question"]
            relevant_ids = set(item["relevant_chunk_ids"])
            
            print(f"\n  問題: {question}")
            print(f"  黃金標準 Chunk IDs: {sorted(list(relevant_ids))}")

            # 分別從兩個知識庫檢索
            primary_docs = primary_retriever.invoke(question)
            print("  --- 從主要知識庫檢索到的區塊 ---")
            if primary_docs:
                for i, doc in enumerate(primary_docs):
                    source_path = os.path.basename(doc.metadata.get('source', 'N/A'))
                    page_num = doc.metadata.get('page', 'N/A')
                    chunk_id = doc.metadata.get('chunk_id', 'N/A')
                    cleaned_content = doc.page_content.replace('\n', ' ').strip()
                    print(f"    主要區塊 {i+1} (ID: {chunk_id}, 來源: {source_path}, 頁碼: {page_num}): \"{cleaned_content[:100]}...\"")
            else:
                print("    未從主要知識庫檢索到任何區塊。")
            
            knowledge_docs = []
            if knowledge_retriever:
                knowledge_docs = knowledge_retriever.invoke(question)
                print("  --- 從補充知識庫檢索到的區塊 ---")
                if knowledge_docs:
                    for i, doc in enumerate(knowledge_docs):
                        source_path = os.path.basename(doc.metadata.get('source', 'N/A'))
                        page_num = doc.metadata.get('page', 'N/A')
                        cleaned_content = doc.page_content.replace('\n', ' ').strip()
                        print(f"    補充區塊 {i+1} (來源: {source_path}, 頁碼: {page_num}): \"{cleaned_content[:100]}...\"")
                else:
                    print("    未從補充知識庫檢索到任何區塊。")
            
            # 合併結果
            retrieved_docs = primary_docs + knowledge_docs
            
            # 從合併後的結果中提取 chunk_id
            # 注意: 補充知識庫的 chunk 沒有 chunk_id，這裡使用 get 避免錯誤
            retrieved_ids = [doc.metadata.get('chunk_id', -1) for doc in retrieved_docs]
            
            # --- 計算 Recall@3 ---
            is_hit = any(ret_id in relevant_ids for ret_id in retrieved_ids)
            recall_scores.append(1.0 if is_hit else 0.0)

            # --- 計算 MRR ---
            reciprocal_rank = 0.0
            for rank, ret_id in enumerate(retrieved_ids, 1):
                if ret_id in relevant_ids:
                    reciprocal_rank = 1.0 / rank
                    break
            mrr_scores.append(reciprocal_rank)

    # 計算最終平均分數
    final_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
    final_mrr = sum(mrr_scores) / len(mrr_scores) if mrr_scores else 0.0
    
    return final_recall, final_mrr

def main():
    """
    主要的執行函式
    """
    try:
        with open('golden_set.json', 'r', encoding='utf-8') as f:
            golden_set = json.load(f)
        print(f"已載入 {len(golden_set)} 個問題樣本。")
    except FileNotFoundError:
        print("錯誤: golden_set.json 不存在。請先建立並完成標記。")
        return
    
    recall, mrr = evaluate_retriever(golden_set, EMBEDDING_MODEL)

    print(f"\n--- Evaluation Results for: {EMBEDDING_MODEL} (Primary + Knowledge) ---")
    print(f"Recall@3: {recall:.4f}")
    print(f"MRR:      {mrr:.4f}")
    print("-------------------------------------------------")

if __name__ == "__main__":
    main()
