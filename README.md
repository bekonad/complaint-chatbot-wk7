# Intelligent Complaint Analysis Chatbot

**10 Academy – KAIM 8 – Week 7 Challenge**  
**Building a RAG-Powered Chatbot to Turn Customer Feedback into Actionable Insights**

**Challenge Period:** 31 Dec 2025 – 13 Jan 2026  
**Repository:** https://github.com/bekonad/complaint-chatbot-wk7  
**Author:** Bereket Feleke (bekonad)

---

## Business Objective & Motivation
CrediTrust Financial is a fast-growing digital finance company serving East African markets through a mobile-first platform. Their offerings span Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.  
With a user base of over 500,000, CrediTrust receives thousands of customer complaints per month. The goal is to build an internal AI tool that transforms unstructured complaint data into actionable insights for stakeholders — reducing analysis time from days to minutes and enabling proactive problem-solving.

**Success KPIs:**
- Reduce time to identify major complaint trends from days to minutes  
- Empower non-technical teams to get answers without data analysts  
- Shift from reactive to proactive issue resolution based on real-time feedback  

---

## Project Structure
```
complaint-chatbot-wk7/
├── data/
│   ├── processed/filtered_complaints.csv     # Task 1 output
│   └── vector_store/chroma_task2_10k/        # Task 2 vector store
├── src/                                      # Scripts for pipeline
├── notebooks/                                # Jupyter notebooks (EDA, evaluation)
├── docs/                                     # Final report, evaluation artifacts
├── figures/                                  # Plots & screenshots
├── app.py                                    # Task 4 Gradio UI
├── requirements.txt                          # Dependencies
├── .gitignore                                # Ignores venv, vector_store, etc.
└── README.md
```

---

## Tasks Completed

### Task 1 – Exploratory Data Analysis & Preprocessing
- Loaded 1.2M rows, analyzed narrative presence and length  
- Filtered to target products + non-empty narratives  
- Cleaned text (lowercase, boilerplate removal, special chars, whitespace normalization)  
- Output: `data/processed/filtered_complaints.csv` (26,168 rows)

### Task 2 – Text Chunking, Embedding & Vector Store Indexing
- Stratified sample of 10,000 complaints  
- Chunking: 500 char size, 50 overlap → 29,667 chunks  
- Embedding: `sentence-transformers/all-MiniLM-L6-v2`  
- Indexed in ChromaDB, persisted to `data/vector_store/chroma_task2_10k`

### Task 3 – RAG Pipeline & Evaluation
- Implemented retriever + generator pipeline with fallback to sample store  
- Prompt template ensures grounded answers only from retrieved complaints  
- Evaluation notebook: `notebooks/03_task3_rag_core_evaluation.ipynb`  
- Artifacts:  
  - `docs/task3_evaluation.md` (Markdown table of 10 questions)  
  - `docs/task3_eval_preview.json` (JSON preview of answers + sources)

### Task 4 – Interactive Chat UI
- Built Gradio interface (`app.py`)  
- Input: natural language question  
- Output: grounded answer + sources (product, issue, excerpt)  
- Screenshot saved in `docs/figures/task4_ui.png`  

---

## Setup Instructions
```bash
git clone https://github.com/bekonad/complaint-chatbot-wk7.git
cd complaint-chatbot-wk7
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run pipeline demo:
```bash
python src/task3_rag_pipeline.py
```

Run evaluation notebook:
- Open `notebooks/03_task3_rag_core_evaluation.ipynb` and run all cells

Launch Gradio UI:
```bash
python app.py
```

---

## References
- [ChromaDB Documentation](https://docs.trychroma.com/)  
- [Sentence Transformers](https://www.sbert.net/)  
- Hugging Face RAG [(huggingface.co in Bing)](https://www.bing.com/search?q="https%3A%2F%2Fhuggingface.co%2Fdocs%2Ftransformers%2Fmain%2Fen%2Ftasks%2Fretrieval")  

---

**Author:** Bereket Feleke  
**Last updated:** 13 Jan 2026