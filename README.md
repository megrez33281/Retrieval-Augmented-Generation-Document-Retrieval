# RAG 問答系統 Demo

這是一個基於「檢索增強生成」(Retrieval-Augmented Generation, RAG) 技術的命令列問答系統  
它能讀取指定的PDF文件作為知識庫，並使用大型語言模型(LLM)來回答使用者針對文件內容提出的問題   


## 功能特色
- 使用`sentence-transformers`進行語意向量化  
- 使用`FAISS`進行高效的向量檢索  
- 整合Google `gemini-2.5-flash`模型生成答案  
- 提供互動式命令列介面(CLI)  

## 環境設定

### Python 版本限制

這個專案的相依套件(特別是`langchain`生態系)與最新的Python版本存在相容性問題。
- **強烈建議使用 `Python 3.9` ~ `Python 3.12` 之間的一個版本。**
- **請勿使用 `Python 3.13` 或更高版本**，否則會導致套件安裝不完整及執行錯誤。

### 安裝步驟

1. **複製專案**
   ```bash
   git clone https://github.com/megrez33281/Retrieval-Augmented-Generation-Document-Retrieval
   cd Retrieval-Augmented-Generation-Document-Retrieval
   ```

2. **建立虛擬環境**
   確保用來執行此指令的`python`是符合版本限制的版本(例如 3.9)。
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
本專案需要存取Google AI的API，因此必須設定`GOOGLE_API_KEY`環境變數。
1. 前往 [Google AI Studio](https://aistudio.google.com/app/apikey) 取得API金鑰。
2. 在 Windows 終端機中執行以下指令來設定環境變數（**設定後需要重新啟動終端機**）：
   ```bash
   setx GOOGLE_API_KEY "API金鑰"
   ```

## 工作流程與使用方法

本專案包含三個主要腳本，其標準工作流程如下：

### 步驟 1: 準備文件

將所有你希望納入主要知識庫的 PDF 檔案放入 `Dataset` 資料夾中  
若有補充知識文件，請放入 `Knowledge` 資料夾中  

### 步驟 2: 產生 Chunk 資料

執行 `prepare_data.py` 腳本  
這個腳本會讀取`Dataset`資料夾中的所有 PDF，將它們切割成帶有全域唯一ID的文本區塊 (chunks)，並將結果儲存為`chunks.json` 
（但是實際要使用的時候只能指定一個主文件進行查詢） 

```bash
# 啟用虛擬環境後執行
python prepare_data.py
```

`chunks.json` 是後續所有流程的單一事實來源，只在此步驟進行文件切割　　

### 步驟 3: 互動式問答

若想與系統進行互動式問答，可以直接執行`rag_baseline.py`　　
腳本內需要指定載入一個PDF檔案作為主要知識庫

```bash
# 啟用虛擬環境後執行
python rag_baseline.py
```
在互動過程中，每次問答的詳細過程也會被記錄到 `qa_log.json` 中  

## 腳本檔案說明

- **`prepare_data.py`**
  - **用途**: 預處理腳本。讀取 `Dataset/` 資料夾中的所有 PDF 檔案，將其內容使用**語意切割器 (Semantic Chunker)** 切割成較小的文本區塊 (chunks)，並儲存為 `chunks.json`。這是使用問答功能的**第一步**。

- **`rag_baseline.py`**
  - **用途**: 核心 RAG 流程與互動式問答介面。
  - **多源檢索**: 同時從主要知識庫 (基於 `chunks.json`) 和補充知識庫 (`Knowledge/` 資料夾中的 PDF) 檢索資訊  
  - **LLM驅動的查詢重寫**: 當檢索結果不足時，LLM會優化問題（要求使用英文）並觸發重新檢索，最多重試 `MAX_RETRIES` 次  
  - **問答日誌**: 將每次問答的詳細過程記錄到 `qa_log.json`中  

- **`evaluate.py`**
  - **用途**: 檢索器性能評估腳本。根據`golden_set.json`中的標準答案，計算指定嵌入模型在`Recall@3`和`MRR`指標上的分數。評估時也會使用多源檢索邏輯  

- **`convert_or_sharc.py`**
  - **用途**: 資料集轉換工具，用於將OR-ShARC資料集（一個用於評估規則匹配能力的標準資料集）轉換為本系統相容的`chunks.json`和`golden_set.json`格式

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
- 採用 **`SemanticChunker`** 進行語意切割  
- 使用 `sentence-transformers/all-MiniLM-L6-v2` 進行語意向量化並建立 FAISS 索引  
- 輸出統一格式之 `chunks.json`，作為後續檢索與評估的單一資料來源  

此資料集同時用於：
1. RAG 問答系統的互動式查詢與展示  
2. 檢索效能量化評估（Recall@k、MRR）  
3. 不同嵌入模型與重排序策略的後續實驗  

## 評估流程

本專案提供了一套標準化的流程來量化評估檢索器 (Retriever) 的性能。

### 步驟 1: 建立黃金標準集

1.  **產生 `chunks.json`**: 確保已經執行過 `prepare_data.py`  
2.  **建立 `golden_set.json`**: 這是一個手動標記的檔案，需要根據`chunks.json` 的內容，為評估問題集填寫正確的`relevant_chunk_ids`。專案中已包含一個範本，可以直接修改或參考其格式

### 步驟 2: 執行評估

當 `golden_set.json` 標記完成後，執行 `evaluate.py` 腳本  

```bash
# 啟用虛擬環境後執行
python evaluate.py
```

腳本會自動為 `golden_set.json` 中的每個來源文件建立專屬的 RAG 流程，並計算 `Recall@3` 和 `MRR` 分數，最後輸出匯總結果  




## 評估結果

### 詳細檢索結果--Baseline  

我們的**Baseline（基線）**採用最簡單的固定長度分塊（brute-force chunking）策略：
  
* **Chunking**：固定長度切割（每1000個字元切割一次，有200個字元的overlap） —— *此為 baseline*
* **Embedding**：`sentence-transformers/all-MiniLM-L6-v2`（baseline embedding）
* **Top-k 檢索**：k = 3（評估 Recall@3 / MRR）
* **評估協議**：使用人工標註的答案集作為標準，所有實驗在相同隨機種子下執行以確保可比較性（但由於Chunking策略不同，產生的Chunk數量也會不同，因此會重新標註答案集） 

  
以下為Baseline執行`evaluate.py`腳本對每個問題進行Chunk檢索的詳細狀況：

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





## 評估結果-語意Chunking
  
### 詳細檢索結果
以下為改進Chunking方式為語意Chunking後執行`evaluate.py`腳本對每個問題進行Chunk檢索的詳細狀況：

| 問題 (縮寫)                                  | 正確答案 (Chunk IDs) | 檢索結果 (Chunk IDs) | 是否命中 |
| -------------------------------------------- | -------------------- | -------------------- | -------- |
| **來源: 114-1 IR Final Project Requirements.pdf** |                      |                      |          |
| What is the total number of students...      | `[0]`                | `[9, 7, 0]`        | True |
| What is the exact submission deadline...     | `[9]`               | `[9, 2, 0]`         | True  |
| List the main components that should be...   | `[2, 3]`          | `[2, 0, 4]`         | True  |
| What percentage of the total grade is...     | `[0]`                | `[0, 7, 9]`         | True  |
| In the final project, is making a demo...    | `[4]`             | `[9, 0, 4]`         | True  |
| **來源: 2025 Generative Information Retrieval HW1.pdf** |                      |                      |          |
| What is the deadline for HW1, and are...     | `[19, 34]`           | `[35, 34, 40]`       | True  |
| For HW1, which two Sparse Retrieval...       | `[19, 24]`               | `[31, 18, 19]`       | True  |
| What is the final scoring metric for...      | `[25]`           | `[23, 28, 25]`       | True  |
| According to the HW1 report submission...    | `[31]`               | `[35, 18, 34]`       | False |
| What is the penalty for failing to...        | `[34]`           | `[35, 34, 30]`       | True  |
  
**分析**:
* 優點
  大多數問題都能成功召回至少一個相關的數據塊 (Recall@3 = 0.9)
  特別是對於日期、截止日期等關鍵字明確的問題，檢索效果很好
* 待改進 
  一個問題檢索失敗，並且這個失敗案例在上一次更換Chunking方式前也失敗了  

   
### 質化分析
為了深入了解檢索失敗的原因，對失敗案例進行質化分析
- **問題**: `"According to the HW1 report submission guidelines, what is the first question that needs to be answered in the report?"`
- **正確答案 Chunk ID**: `[31]`
- **系統檢索到的 Chunk IDs**: `[35, 18, 34]`
#### 錯誤分析  
- **正確的 Chunk (ID 31)**:
  包含了明確的答案`"Report Submission\nAnswer the following 3 questions:..."`  
  並且其實蠻明顯的
- **錯誤檢索的 Chunk (ID 35, 18, 34)**:
  這些區塊或多或少提到了`question`、`ubmission`、`HW1`等關鍵字，但實際上完全不相關  
  推測還是這個問題對於語意的理解要求比較嚴格  



## 進階評估：模型比較

為了更深入地評估檢索器的核心性能，本專案引入了OR-ShARC資料集，並建立了一套自動化比較不同嵌入模型的流程。

### 1. OR-ShARC 資料集評估

OR-ShARC 是一個專門用於評估「規則遵循問答」的資料集。
它的特點是問題通常需要精準匹配到知識庫中的特定規則條文才能正確回答。
這使它成為測試嵌入模型對長篇、結構化文本語義理解能力的絕佳基準。
不過鑒於其中的資料屬於"已經被切割過的"，此處只用此資料集評估在已經切好Chunk的前提下，是否能準確匹配相應的區塊的能力。

我們使用`dev`集作為驗證資料，對比了兩個常用的嵌入模型。

#### 評估結果比較
Total Samples Evaluated: 1105
Total Chunks in Knowledge Base: 651

| Model                                         | Recall@3 | MRR      |
| --------------------------------------------- | -------- | -------- |
| `sentence-transformers/all-MiniLM-L6-v2`      | 0.8244   | 0.7195   |
| `sentence-transformers/all-mpnet-base-v2`     | 0.8326   | 0.7311   |

**結果分析**:
- **`all-mpnet-base-v2` 表現更優**：無論是 `Recall@3` 還是 `MRR`，`all-mpnet-base-v2` 的分數都更高，證明它在理解規則文本的細微語義差異上，比輕量的 `all-MiniLM-L6-v2` 更具優勢。
- **驗證了系統有效性**：兩個模型都取得了不錯的成績（Recall@3 > 0.82），證明目前的 RAG 架構在處理特定領域的檢索任務時是有效且可靠的。

### 2. 自動化模型比較流程

為了方便進行模型比較，專案新增了以下腳本與功能：

- **`convert_or_sharc.py`**: 用於將原始的 OR-ShARC 資料集轉換為本系統相容的`chunks.json`和`golden_set.json`格式。
- **`compare_models.py`**: 自動化評估腳本。你可以在此腳本的`MODELS_TO_COMPARE`列表中加入多個`sentence-transformers`模型名稱，它會自動為每個模型執行完整的評估流程，並在最後生成清晰的比較表格。

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

長文本問答任務中，RAG 系統的關鍵在於「如何有效地從原始文件中檢索出與問題最相關的內容」。
本研究的重點在於**檢索階段的語義精度提升**，因此回顧主要分為三個方向：**語義分塊（Semantic Chunking）**、**重排序（Re-ranking）** 與 **嵌入模型選型（Embedding Selection）**。

---

### 1. 語義分塊與動態切割 (Semantic/Dynamic Chunking)

Sheng 等人 [1] 提出 **Dynamic Chunking and Selection** 方法，根據句子間的語義相似度動態決定切割邊界，
並使用question-aware篩選器保留最相關區塊，有效避免了傳統固定長度切割造成的語義稀釋問題。
本研究採用類似思路，嘗試改善chunk的語義一致性，預期能提升檢索階段的精確度與上下文完整性。

---

### 2. 檢索結果重排序 (Re-ranking with Cross-Encoders)

Nogueira 與 Cho [2] 提出利用 **BERT Cross-Encoder** 進行 passage re-ranking，
在初步召回後重新評估每個`(query, passage)`的語意匹配度。
此方法能顯著提升top-1命中率與整體MRR，是現今許多RAG系統採用的標準做法。
本研究未來計畫在FAISS檢索之後，導入cross-encoder re-ranking以提升高語義需求問題的精確性。

---

### 3. 嵌入模型選型與語義對齊 (Embedding Model Selection)

Muennighoff 等人 [3] 的 **MTEB Benchmark** 系統性比較了多種sentence-transformer模型在檢索與語義任務上的表現，
結果顯示`all-mpnet-base-v2`、`E5-large`等模型在語義檢索上普遍優於輕量模型`MiniLM-L6-v2`。
本研究據此計畫評估不同embedding模型對Recall@k與MRR的影響，以尋找性能與效率的最佳平衡點。

---

### 參考文獻

[1] Y. Sheng, S. Liu, & R. Zhao, “Dynamic chunking and selection for reading comprehension of ultra-long context in large language models,” *Proc. 63rd Annual Meeting of the ACL*, 2025.
[2] R. Nogueira & K. Cho, “Passage re-ranking with BERT,” *arXiv preprint* arXiv:1901.04085, 2019.
[3] N. Muennighoff et al., “MTEB: Massive Text Embedding Benchmark,” *NeurIPS Datasets and Benchmarks Track*, 2023.


## 進階功能  

### 1. 多源知識檢索  
系統同時支援從兩個知識來源進行檢索：
- **主要知識庫**: 來自使用者想要行提問的作業/專案說明
- **補充知識庫**: 來自使用者提供的作業/專案相關的知識（將相關PDF檔案放置到資料夾中）
LLM 會綜合這兩種知識來源來生成答案  

### 2. LLM驅動的查詢重寫 (LLM-Driven Query Rewriting)
當 LLM 判斷當前檢索到的資訊不足以回答問題時，它會觸發查詢重寫機制：
- **觸發條件**: LLM 在回答中輸出 `[QUERY_REWRITE]` 特殊標記，並附帶一個優化後的問題   
- **優化問題語言**: 優化後的問題會被要求使用英文，以提高嵌入模型的檢索效果（目前測試文件會是英文）
- **重試次數**: 系統最多會進行 `MAX_RETRIES` (目前設定為 2) 次重寫與重新檢索　　


### ３. 問答日誌記錄 (QA Logging)
每次互動式問答的詳細過程都會被記錄到 `qa_log.json` 檔案中：
- **記錄內容**: 包含原始問題、LLM 生成的最終答案，以及所有參考來源（文件名稱和頁碼）　　
- **用途**: 便於開發者觀察系統行為、分析檢索與生成效果，並作為未來改進的數據基礎　　

### 5. 語意分塊 (Semantic Chunking)
將原先的Chunk策略改變為基於語意進行Chunking  
此方法根據文本的語意相似度來決定切割點，旨在生成語意更連貫、更完整的文本區塊  
並且新的Chunking策略在Recall@3的成績上升至0.9000

## 未來優化方向

目前的 RAG 系統是一個良好的基線，但有許多具有泛化能力的優化方向值得探索，以進一步提升系統性能：

1.  **更換嵌入模型 (Embedding Model)**
    - 目前使用的是輕量的 `all-MiniLM-L6-v2`。更換為更強大、更深層的模型（如 `all-mpnet-base-v2` 或其他 MTEB 排行榜上領先的模型）可能會顯著提升對語意細微差別的捕捉能力，從而提高檢索準確率。

2.  **查詢擴展 (Query Expansion)**
    - 在檢索前，可以利用 LLM 將使用者的原始問題改寫或擴展成多個語意相近的問題（例如，"小組人數" -> "how many students per group", "group size"）。使用多個問題進行搜索可以覆蓋更多樣的文本表述，提高召回率。

3.  **重排序 (Re-ranking)**
    - 在檢索器召回 top-k 個文件後（例如 k=20），可以再引入一個更精準但計算量更大的「重排序模型」（如 Cross-encoder）。這個模型會將問題與每個召回的 chunk 進行成對比較，並給出更精準的相關性分數，然後重新排序，將最相關的結果排到最前面。


## 注意事項

為了方便進行評估，本專案進行了特殊的架構設計：
- `prepare_data.py` 會對`Dataset/`目錄下的全部PDF檔案進行Chunking，並合併成一個大型的`chunks.json`，其中的`chunk_id` 是全域唯一的。
- 在呼叫`evaluate.py`時，腳本會根據`golden_set.json`中標註的`source_file`，從 `chunks.json` 篩選出對應來源的chunk，並為其建立一個獨立的、隔離的向量資料庫 (`faiss_index_*`) 來進行評估  
- `rag_baseline.py` 為單一文件建立獨立的RAG流程。需要在其中指定目標文件，才能對目標文件行檢索


## 相關問題  

1. 把簡報(或課程資料)丟到GPT-5或Gemini 2.5，模型是否能夠正確回答   
  如果單純的將目前的課程文件丟給Gemini 2.5，他也可以很好的回答相關的問題  
  不過在經過新一輪的功能擴充後，目前的系統可以允許使用者將一些背景知識的PDF檔（例如課堂知識）丟到`Knowledge`資料夾中　　
  系統在檢索的時候，會將Knowledge資料夾中的PDF檔案也進行Chunking，並且在使用者提問時，也會從這個Chunking中檢索最相關的幾個Chunk一起提供給LLM，增強回答的品質（由於是根據使用者提供的知識文件，模型的解答會更貼近使用者學習的知識）　　

  * 最後，開發此系統的目的，其實不只是為了增強模型回答的品質，還有：
    1. 降低Gemini 2.5等大語言模型出現幻覺的機率  
    2. 降低使用Gemini 2.5等大語言模型的成本（避免直接將文件丟給模型的高昂COST） 
    3. 就理論而言，一些免費版的大語言模型沒辦法吃的大檔案，可以使用此系統進行提問  


2. 作業要求的檢索，動機強調了要自建，好奇是否能從別人公開的作業收集來進行
  可以，這個系統開發的最初就定下了要能夠盡量的泛用的基調，因此相關的優化時會特意避免微調這類比較針對性的優化手段  
  所以雖然沒有測試，但只要作業內容、格式不要太特殊、太奇怪，通常都能使用  

3. 資料集的標註怎麼進行的? 方法看起來就是原始 RAG，好奇有沒有對實作完成後的表現做分析，目前沒有比較的 baseline 不知道到底 recall@3 0.8 算不算好
  由於使用的資料集是沒有在市面上公開的作業說明，因此此處會先利用Chunk的method將相關資料進行Chunking，由我們自行擬定一些問題，然後人工在這些切出的Chunk中尋找正確答案（所以每次優化Chunking方式都要重新標註......）  
  測試的時候，則會用匹配出的Chunk去對比答案，計算Recall@3以及MRR  
  至於Baseline則只有針對內部進行比較，即優化前與優化後的表現比較　　
  


