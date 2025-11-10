
import os
import json
from rag_baseline import setup_rag_pipeline, EMBEDDING_MODEL

def evaluate_retriever(golden_set, embedding_model):
    """
    使用黃金標準集和指定的嵌入模型來評估檢索器的 Recall@3 和 MRR。
    """
    recall_scores = []
    mrr_scores = []
    
    # 按來源文件將問題分組，避免重複為同一個檔案建立 RAG pipeline
    questions_by_source = {}
    for item in golden_set:
        # 確保所有 relevant_chunk_ids 都被填寫了
        if not item["relevant_chunk_ids"]:
            print(f"警告: 問題 \"{item['question'][:30]}...\" 的 relevant_chunk_ids 為空，將跳過此問題的評估。")
            continue
        
        source_filename = item["source_file"] # 直接使用檔名作為 key
        if source_filename not in questions_by_source:
            questions_by_source[source_filename] = []
        questions_by_source[source_filename].append(item)

    if not questions_by_source:
        print("錯誤: 沒有可供評估的問題。請檢查 golden_set.json 是否已正確標記。")
        return 0.0, 0.0

    print(f"\n--- 使用模型: {embedding_model} ---")
    print(f"開始評估 {len(golden_set)} 個問題，來源分為 {len(questions_by_source)} 個檔案...")

    # 為每個來源檔案建立一次 RAG pipeline
    for source_filename, questions in questions_by_source.items():
        print(f"\n--- 正在處理來源: {source_filename} ---")
        # 為這個來源檔案和模型建立專屬的 retriever
        pipeline_components = setup_rag_pipeline(
            source_file_name=source_filename,
            embedding_model=embedding_model
        )
        retriever = pipeline_components["retriever"]

        # 在這個 retriever 上評估所有相關問題
        for item in questions:
            question = item["question"]
            relevant_ids = set(item["relevant_chunk_ids"])
            
            retrieved_docs = retriever.invoke(question)
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
            
            # 減少輸出，只在出錯時顯示詳細資訊
            # print(f"  Q: {question[:40]}...")
            # print(f"     - Ground Truth IDs: {sorted(list(relevant_ids))}")
            # print(f"     - Retrieved IDs:    {retrieved_ids}")
            # print(f"     - Hit: {is_hit}, RR: {reciprocal_rank:.2f}")

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
    except FileNotFoundError:
        print("錯誤: golden_set.json 不存在。請先建立並完成標記。")
        return
    
    recall, mrr = evaluate_retriever(golden_set, EMBEDDING_MODEL)

    print("\n--- 最終評估結果 ---")
    print(f"模型: {EMBEDDING_MODEL}")
    print(f"Recall@3: {recall:.4f}")
    print(f"MRR:      {mrr:.4f}")
    print("--------------------")

if __name__ == "__main__":
    main()
