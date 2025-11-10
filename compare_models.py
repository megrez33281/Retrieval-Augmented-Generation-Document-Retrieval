
import json
from evaluate import evaluate_retriever

# --- Models to Compare ---
MODELS_TO_COMPARE = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
]

def main():
    """
    Main function to run the model comparison.
    """
    print("--- Starting Model Comparison ---")

    try:
        with open('golden_set.json', 'r', encoding='utf-8') as f:
            golden_set = json.load(f)
        print(f"Loaded {len(golden_set)} questions from golden_set.json")
        with open('chunks.json', 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks from chunks.json")
    except FileNotFoundError as e:
        print(f"Error: {e.filename} not found. Please make sure it has been created.")
        return

    results = []
    for model in MODELS_TO_COMPARE:
        recall, mrr = evaluate_retriever(golden_set, model)
        results.append({
            "model": model,
            "recall@3": recall,
            "mrr": mrr
        })

    # --- Print Final Comparison Table ---
    print("\n\n--- Final Model Comparison Results ---")
    print(f"Total Samples Evaluated: {len(golden_set)}")
    print(f"Total Chunks in Knowledge Base: {len(chunks)}")
    # Print header
    print(f"{'Model':<45} | {'Recall@3':<10} | {'MRR':<10}")
    print("-" * 70)
    # Print results
    for res in results:
        print(f"{res['model']:<45} | {res['recall@3']:.4f}     | {res['mrr']:.4f}")
    print("----------------------------------------")

if __name__ == "__main__":
    main()
