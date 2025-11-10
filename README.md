# RAG 問答系統 Demo

這是一個基於「檢索增強生成」(Retrieval-Augmented Generation, RAG) 技術的命令列問答系統  
它能讀取指定的PDF文件作為知識庫，並使用大型語言模型(LLM)來回答使用者針對文件內容提出的問題  


## 功能特色

- 讀取PDF文件作為知識庫  
- 使用`sentence-transformers`進行語意向量化  
- 使用`FAISS`進行高效的向量檢索  
- 整合Google `gemini-2.5-flash`模型生成答案  
- 提供互動式命令列介面(CLI)  

## 環境設定

### Python 版本限制

這個專案的相依套件(特別是`langchain`生態系)與最新的Python版本存在相容性問題  
- **強烈建議使用 `Python 3.9` ~ `Python 3.12` 之間的一個版本。**
- **請勿使用 `Python 3.13` 或更高版本**，否則會導致套件安裝不完整及執行錯誤  

### 安裝步驟

1. **複製專案**
   ```bash
   git clone https://github.com/megrez33281/Retrieval-Augmented-Generation-Document-Retrieval
   cd Retrieval-Augmented-Generation-Document-Retrieval
   ```  

2. **建立虛擬環境**
   確保用來執行此指令的`python`是符合版本限制的版本(例如 3.9)  
   ```bash
   python -m venv venv
   ```

3. **啟用虛擬環境**
   - 在 Windows (CMD/PowerShell) 上:  
     ```bash
     .\venv\Scripts\activate
     ```
   - 在 macOS/Linux 上:
     ```bash
     source venv/bin/activate
     ```

4. **安裝相依套件**
   ```bash
   pip install -r requirements.txt
   ```

## API 金鑰設定
本專案需要存取Google AI的API，因此必須設定`GOOGLE_API_KEY`環境變數  
1. 前往 [Google AI Studio](https://aistudio.google.com/app/apikey) 取得API金鑰  
2. 在 Windows 終端機中執行以下指令來設定環境變數（**設定後需要重新啟動終端機**）
   ```bash
   setx GOOGLE_API_KEY "API金鑰"  
   ```

## 工作流程與使用方法

本專案包含三個主要腳本，其標準工作流程如下：

### 步驟 1: 準備文件

將所有你希望納入知識庫的 PDF 檔案放入 `Dataset` 資料夾中。

### 步驟 2: 產生 Chunk 資料

執行 `prepare_data.py` 腳本。這個腳本會讀取 `Dataset` 資料夾中的所有 PDF，將它們切割成帶有全域唯一 ID 的文本區塊 (chunks)，並將結果儲存為 `chunks.json`。

```bash
# 啟用虛擬環境後執行
python prepare_data.py
```

`chunks.json` 是後續所有流程的「單一事實來源」，只在此步驟進行文件切割。

### 步驟 3: 互動式問答

若想與特定文件進行互動式問答，可以直接執行 `rag_baseline.py`。腳本內預設會載入 `114-1 IR Final Project Requirements.pdf`。

```bash
# 啟用虛擬環境後執行
python rag_baseline.py
```

## 腳本檔案說明

- **`prepare_data.py`**
  - **用途**: 預處理腳本。讀取 `Dataset/` 資料夾中的所有 PDF 檔案，將其內容切割成較小的文本區塊 (chunks)，並儲存為 `chunks.json`。這是使用問答功能的**第一步**  
  
- **`rag_baseline.py`**
  - **用途**: 核心 RAG 流程與互動式問答介面。此腳本包含建立檢索器 (Retriever) 和生成鏈 (Generation Chain) 的主要邏輯 (`setup_rag_pipeline`)。直接執行此腳本，可以針對單一PDF文件進行互動式問答  

- **`evaluate.py`**
  - **用途**: 檢索器性能評估腳本。根據`golden_set.json`中的標準答案，計算指定嵌入模型在`Recall@3`和`MRR`指標上的分數  

- **`convert_or_sharc.py`**
  - **用途**: 資料集轉換工具。用於將OR-ShARC資料集（一個用於評估規則匹配能力的標準資料集）轉換為本系統相容的`chunks.json`和`golden_set.json`格式。  

- **`compare_models.py`**
  - **用途**: 自動化模型比較腳本。可以一次評估`MODELS_TO_COMPARE`列表中定義的多個嵌入模型，並在最後輸出所有模型的性能比較表，方便進行分析  

## 資料集說明 (Dataset)
本專案測試時使用的知識庫由兩份Generative Information Retrieval課程文件組成：
  
| 文件名稱 | 類型 | 主要內容 |
|-----------|------|-----------|
| **GIR HW1** | 作業說明 | 定義 Sparse / Dense retrieval、截止日期與評分方式 |
| **GIR Project Requirements** | 專案說明 | 描述期末專案規範、分組與評分比例 |

所有文件皆放置於 `Dataset/` 資料夾中，並透過 `prepare_data.py` 進行以下預處理：  
- 使用 `PyMuPDF` 擷取文字並移除頁眉、頁碼與多餘換行  
- 採用 `RecursiveCharacterTextSplitter` 以 `chunk_size=1000`、`overlap=200` 進行切割  
- 使用 `sentence-transformers/all-MiniLM-L6-v2` 進行語意向量化並建立 FAISS 索引  
- 輸出統一格式之 `chunks.json`，作為後續檢索與評估的單一資料來源  

此資料集同時用於：  
1. RAG 問答系統的互動式查詢與展示  
2. 檢索效能量化評估（Recall@k、MRR）  
3. 不同嵌入模型與重排序策略的後續實驗
  

## 評估流程

本專案提供了一套標準化的流程來量化評估檢索器 (Retriever) 的性能。

### 步驟 1: 建立黃金標準集

1.  **產生 `chunks.json`**: 確保你已經執行過 `prepare_data.py`。
2.  **建立 `golden_set.json`**: 這是一個手動標記的檔案，你需要根據 `chunks.json` 的內容，為你的評估問題集填寫正確的 `relevant_chunk_ids`。專案中已包含一個範本，你可以直接修改或參考其格式。

### 步驟 2: 執行評估

當 `golden_set.json` 標記完成後，執行 `evaluate.py` 腳本。

```bash
# 啟用虛擬環境後執行
python evaluate.py
```

腳本會自動為 `golden_set.json` 中的每個來源文件建立專屬的 RAG 流程，並計算 `Recall@3` 和 `MRR` 分數，最後輸出匯總結果。

## 評估結果

根據 `golden_set.json` 的標準答案，目前的 RAG 系統在基準設定下（`all-MiniLM-L6-v2` 嵌入模型）的檢索性能如下：

- **Recall@3:** 0.8000
- **MRR:** 0.5667

### 詳細檢索結果  
以下為執行`evaluate.py`腳本對每個問題進行Chunk檢索的詳細狀況：

| 問題 (縮寫)                                  | 正確答案 (Chunk IDs) | 檢索結果 (Chunk IDs) | 是否命中 |
| -------------------------------------------- | -------------------- | -------------------- | -------- |
| **來源: 114-1 IR Final Project Requirements.pdf** |                      |                      |          |
| What is the total number of students...      | `[0]`                | `[11, 1, 15]`        | False |
| What is the exact submission deadline...     | `[15]`               | `[15, 1, 5]`         | True  |
| List the main components that should be...   | `[3, 4, 5]`          | `[14, 1, 3]`         | True  |
| What percentage of the total grade is...     | `[1]`                | `[1, 0, 11]`         | True  |
| In the final project, is making a demo...    | `[5, 6]`             | `[15, 6, 5]`         | True  |
| **來源: 2025 Generative Information Retrieval HW1.pdf** |                      |                      |          |
| What is the deadline for HW1, and are...     | `[31, 41]`           | `[41, 38, 30]`       | True  |
| For HW1, which two Sparse Retrieval...       | `[31]`               | `[39, 30, 31]`       | True  |
| What is the final scoring metric for...      | `[35, 37]`           | `[40, 37, 41]`       | True  |
| According to the HW1 report submission...    | `[39]`               | `[41, 30, 38]`       | False |
| What is the penalty for failing to...        | `[38, 41]`           | `[41, 38, 30]`       | True  |  
**Recall@3:** 0.8000  
**MRR:** 0.5667          
- **分析**:
  - **優點**  
   大多數問題都能成功召回至少一個相關的數據塊 (Recall@3 = 0.8)  
   特別是對於日期、截止日期等關鍵字明確的問題，檢索效果很好  
  - **待改進**
   有兩個問題檢索失敗。  
   例如問題"What is the total number of students..."，其答案在文件開頭，但系統召回了其他看似相關但無關的區塊  
   這可能表示目前的嵌入模型對於某些語意細節的捕捉能力有限，或是文本切割策略可以再優化  

### 質化分析
為了深入了解檢索失敗的原因，對失敗案例進行質化分析  
- **問題**: `"What is the total number of students allowed in a single group for the final project?"`
- **正確答案 Chunk ID**: `[0]`
- **系統檢索到的 Chunk IDs**: `[11, 1, 15]`
#### 1. 內容比對  
- **正確的 Chunk (ID 0)**:  
  包含了明確的答案`"...you will work in groups of 3 ~ 4 students..."`。  
- **錯誤檢索的 Chunk (ID 11, 1, 15)**:  
  這些區塊雖然也提到了"final project"，但其核心內容分別是關於「成員貢獻」、「評分標準」和「報告提交者」，並未涉及「小組人數」。

#### 2. 錯誤分析 (Error Analysis)
這次的失敗是典型的**檢索錯誤 (Retrieval Error)**。  
系統成功匹配了 "final project" 這個關鍵詞，但未能準確理解問題的核心語意——「小組人數」  
**可能的原因假設：**
1.  **語意理解粒度不足**:
  all-MiniLM-L6-v2` 模型可能無法精細地區分與「專案」相關的不同主題（如人數、貢獻、評分），導致它召回了任何看似相關的區塊。
2.  **Chunk 語意被稀釋**: 
  正確答案所在的`Chunk 0` 內容較為混雜，除了人數資訊外，還包含了專案主題介紹。  
  這可能導致其整體的語意向量被「稀釋」，在向量空間中反而不如那些主題更單一的錯誤區塊與問題來得接近。


## 進階評估：模型比較

為了更深入地評估檢索器的核心性能，本專案引入了OR-ShARC資料集，並建立了一套自動化比較不同嵌入模型的流程   

### 1. OR-ShARC 資料集評估
  
OR-ShARC 是一個專門用於評估「規則遵循問答」的資料集  
它的特點是問題通常需要精準匹配到知識庫中的特定規則條文才能正確回答  
這使它成為測試嵌入模型對長篇、結構化文本語義理解能力的絕佳基準
不過鑒於其中的資料屬於"已經被切割過的"，此處只用此資料集評估在已經切好Chunk的前提下，是否能準確匹配相應的區塊的能力  

我們使用`dev`集作為驗證資料，對比了兩個常用的嵌入模型  

#### 評估結果比較  
Total Samples Evaluated: 1105
Total Chunks in Knowledge Base: 651
     
| Model                                         | Recall@3 | MRR      |
| --------------------------------------------- | -------- | -------- |
| `sentence-transformers/all-MiniLM-L6-v2`      | 0.8244   | 0.7195   |
| `sentence-transformers/all-mpnet-base-v2`     | 0.8326   | 0.7311   |

**結果分析**:
- **`all-mpnet-base-v2` 表現更優**：無論是 `Recall@3` 還是 `MRR`，`all-mpnet-base-v2` 的分數都更高，證明它在理解規則文本的細微語義差異上，比輕量的 `all-MiniLM-L6-v2` 更具優勢  
- **驗證了系統有效性**：兩個模型都取得了不錯的成績（Recall@3 > 0.82），證明目前的 RAG 架構在處理特定領域的檢索任務時是有效且可靠的  

### 2. 自動化模型比較流程

為了方便進行模型比較，專案新增了以下腳本與功能：
  
- **`convert_or_sharc.py`**: 用於將原始的 OR-ShARC 資料集轉換為本系統相容的`chunks.json`和`golden_set.json`格式  
- **`compare_models.py`**: 自動化評估腳本。你可以在此腳本的`MODELS_TO_COMPARE`列表中加入多個`sentence-transformers`模型名稱，它會自動為每個模型執行完整的評估流程，並在最後生成清晰的比較表格  

#### 使用方法
1. **準備資料** (僅需執行一次)
   - 將 OR-ShARC 資料集解壓縮至 `Dataset_OR-ShARC` 資料夾。
   - 執行轉換腳本：
     ```bash
     python convert_or_sharc.py
     ```

2. **執行比較**
   - （可選）在 `compare_models.py` 中編輯 `MODELS_TO_COMPARE` 列表。
   - 執行比較腳本：
     ```bash
     python compare_models.py
     ```


## 文獻回顧（Literature Review / Related Work）

長文本問答任務中，RAG 系統的關鍵在於「如何有效地從原始文件中檢索出與問題最相關的內容」  
本研究的重點在於**檢索階段的語義精度提升**，因此回顧主要分為三個方向：**語義分塊（Semantic Chunking）**、**重排序（Re-ranking）** 與 **嵌入模型選型（Embedding Selection）**  

---

### 1. 語義分塊與動態切割 (Semantic/Dynamic Chunking)

Sheng 等人 [1] 提出 **Dynamic Chunking and Selection** 方法，根據句子間的語義相似度動態決定切割邊界，
並使用question-aware篩選器保留最相關區塊，有效避免了傳統固定長度切割造成的語義稀釋問題  
本研究採用類似思路，嘗試改善chunk的語義一致性，預期能提升檢索階段的精確度與上下文完整性   

---

### 2. 檢索結果重排序 (Re-ranking with Cross-Encoders)

Nogueira 與 Cho [2] 提出利用 **BERT Cross-Encoder** 進行 passage re-ranking，
在初步召回後重新評估每個`(query, passage)`的語意匹配度  
此方法能顯著提升top-1命中率與整體MRR，是現今許多RAG系統採用的標準做法  
本研究未來計畫在FAISS檢索之後，導入cross-encoder re-ranking以提升高語義需求問題的精確性  

---

### 3. 嵌入模型選型與語義對齊 (Embedding Model Selection)

Muennighoff 等人 [3] 的 **MTEB Benchmark** 系統性比較了多種sentence-transformer模型在檢索與語義任務上的表現，
結果顯示`all-mpnet-base-v2`、`E5-large`等模型在語義檢索上普遍優於輕量模型`MiniLM-L6-v2`  
本研究據此計畫評估不同embedding模型對Recall@k與MRR的影響，以尋找性能與效率的最佳平衡點  

---

### 參考文獻

[1] Y. Sheng, S. Liu, & R. Zhao, “Dynamic chunking and selection for reading comprehension of ultra-long context in large language models,” *Proc. 63rd Annual Meeting of the ACL*, 2025.
[2] R. Nogueira & K. Cho, “Passage re-ranking with BERT,” *arXiv preprint* arXiv:1901.04085, 2019.
[3] N. Muennighoff et al., “MTEB: Massive Text Embedding Benchmark,” *NeurIPS Datasets and Benchmarks Track*, 2023.



## 未來優化方向

目前的 RAG 系統是一個良好的基線，但有許多具有泛化能力的優化方向值得探索，以進一步提升系統性能：

1.  **更換嵌入模型 (Embedding Model)**
    - 目前使用的是輕量的 `all-MiniLM-L6-v2`。更換為更強大、更深層的模型（如 `all-mpnet-base-v2` 或其他 MTEB 排行榜上領先的模型）可能會顯著提升對語意細微差別的捕捉能力，從而提高檢索準確率。

2.  **調整文本切割策略 (Chunking Strategy)**
    - 嘗試改用語意進行Chunk切割，讓模型使用的Chunk與問題更加相關  

3.  **查詢擴展 (Query Expansion)**
    - 在檢索前，可以利用 LLM 將使用者的原始問題改寫或擴展成多個語意相近的問題（例如，"小組人數" -> "how many students per group", "group size"）。使用多個問題進行搜索可以覆蓋更多樣的文本表述，提高召回率。

4.  **重排序 (Re-ranking)**
    - 在檢索器召回 top-k 個文件後（例如 k=20），可以再引入一個更精準但計算量更大的「重排序模型」（如 Cross-encoder）。這個模型會將問題與每個召回的 chunk 進行成對比較，並給出更精準的相關性分數，然後重新排序，將最相關的結果排到最前面。


## 注意事項

為了方便進行評估，本專案進行了特殊的架構設計：
- `prepare_data.py` 會分割 `Dataset/` 目錄下的全部 PDF 檔案，並合併成一個大型的 `chunks.json`，其中的 `chunk_id` 是全域唯一的。
- 在呼叫 `evaluate.py` 時，腳本會根據 `golden_set.json` 中標註的 `source_file`，從 `chunks.json` 篩選出對應來源的 chunk，並為其建立一個獨立的、隔離的向量資料庫 (`faiss_index_*`) 來進行評估。
- `rag_baseline.py` 在單獨執行時，也是為單一文件建立獨立的 RAG 流程。若要更換目標文件，需修改其 `main` 區塊中的 `pdf_to_query_name` 變數。