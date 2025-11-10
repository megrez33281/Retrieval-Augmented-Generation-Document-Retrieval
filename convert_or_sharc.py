
import json
import os

# --- Configuration ---
INPUT_DIR = "Dataset_OR-ShARC"
ID2SNIPPET_FILE = "id2snippet.json"
DEV_FILE = "open_retrieval_sharc_dev.json"

OUTPUT_CHUNKS_FILE = "chunks.json"
OUTPUT_GOLDEN_SET_FILE = "golden_set.json"

# --- Main Conversion Logic ---

def convert_id2snippet_to_chunks():
    """
    Loads id2snippet.json and converts it into the RAG-compatible chunks.json format.
    """
    input_path = os.path.join(INPUT_DIR, ID2SNIPPET_FILE)
    print(f"Loading snippets from {input_path}...")
    
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            id2snippet = json.load(f)
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Please make sure the dataset is in the correct directory.")
        return False

    chunks_data = []
    for snippet_id, content in id2snippet.items():
        # Ensure compatibility with evaluate.py, which expects 'content' and 'metadata'
        chunk = {
            "chunk_id": int(snippet_id),
            "content": content,
            "metadata": {"source": ID2SNIPPET_FILE} # Use filename as source
        }
        chunks_data.append(chunk)
        
    print(f"Writing {len(chunks_data)} chunks to {OUTPUT_CHUNKS_FILE}...")
    with open(OUTPUT_CHUNKS_FILE, 'w', encoding='utf-8') as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=4)
        
    print("chunks.json has been created successfully.")
    return True

def convert_dev_to_golden_set():
    """
    Loads dev.json and converts it into the RAG-compatible golden_set.json format.
    """
    input_path = os.path.join(INPUT_DIR, DEV_FILE)
    print(f"Loading development set from {input_path}...")

    dev_data = []
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                dev_data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: {input_path} not found. Please make sure the dataset is in the correct directory.")
        return False

    golden_data = []
    for item in dev_data:
        # Combine question and scenario as per instructions
        query = item.get("question", "") + " " + item.get("scenario", "")
        
        # Get the gold snippet ID
        gold_id = item.get("gold_snippet_id")
        
        if gold_id is None:
            continue

        # Ensure compatibility with evaluate.py, which expects 'source_file'
        golden_record = {
            "question": query.strip(),
            "source_file": ID2SNIPPET_FILE, # Must match the source in chunks.json metadata
            "relevant_chunk_ids": [int(gold_id)]
        }
        golden_data.append(golden_record)

    print(f"Writing {len(golden_data)} records to {OUTPUT_GOLDEN_SET_FILE}...")
    with open(OUTPUT_GOLDEN_SET_FILE, 'w', encoding='utf-8') as f:
        json.dump(golden_data, f, ensure_ascii=False, indent=4)
        
    print("golden_set.json has been created successfully.")
    return True

if __name__ == "__main__":
    print("--- Starting OR-ShARC Dataset Conversion ---")
    


    if convert_id2snippet_to_chunks():
        if convert_dev_to_golden_set():
            print("\n--- Conversion Complete ---")
            print("You can now run the evaluation script:")
            print("python evaluate.py")
    else:
        print("\n--- Conversion Failed ---")
