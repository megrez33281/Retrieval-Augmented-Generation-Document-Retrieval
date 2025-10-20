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

## 使用方法
完成所有設定後，直接執行主程式：
```bash
python rag_baseline.py
```
程式啟動後會先載入或建立向量資料庫，然後回答一個內建的範例問題  
之後，就可以在命令列中輸入自己的問題並按Enter提問  

若要結束程式，輸入`exit`並按Enter  


## 客製化與調整
可以在 `rag_baseline.py`檔案的開頭處，調整以下超參數：
- `PDF_PATH`: 更換您想作為知識庫的 PDF 檔案路徑  
- `EMBEDDING_MODEL`: 更換不同的 `sentence-transformers` 嵌入模型  
- `LLM_MODEL`: 更換不同的 Google AI 模型 (例如 `gemini-2.5-pro`)  
- `CHUNK_SIZE` / `CHUNK_OVERLAP`: 調整文件分塊的大小與重疊字數
- `RETRIEVER_K`: 調整每次檢索時，要回傳給LLM的最相關文件區塊數量 
