# Intelligent Complaint Analysis Chatbot

**10 Academy – KAIM 8 – Week 7 Challenge**  
**Building a RAG-Powered Chatbot to Turn Customer Feedback into Actionable Insights**

**Challenge Period:** 31 Dec 2025 – 13 Jan 2026  
**Current Progress Date:** 3 January 2026  
**Repository:** https://github.com/bekonad/complaint-chatbot-wk7  
**Author:** bekonad (all tasks implemented by me)

## Business Objective & Motivation

CrediTrust Financial is a fast-growing digital finance company serving East African markets through a mobile-first platform. Their offerings span:

- Credit Cards
- Personal Loans
- Savings Accounts
- Money Transfers

With a user base of over 500,000 and operations expanding into three countries, CrediTrust receives thousands of customer complaints per month through in-app channels, email, and regulatory reporting portals.

The goal is to build an internal AI tool that transforms unstructured complaint data into actionable insights for stakeholders like Asha (Product Manager – Credit Cards), Support, Compliance, and Executives — reducing analysis time from days to minutes and enabling proactive problem-solving.

**Success KPIs:**
- Reduce time to identify major complaint trends from days to minutes
- Empower non-technical teams to get answers without data analysts
- Shift from reactive to proactive issue resolution based on real-time feedback

## Project Structure (as implemented)

complaint-chatbot-wk7/
├── data/
│   ├── processed/
│   │   └── filtered_complaints.csv          # Task 1: 26,168 cleaned complaints
│   └── vector_store/
│       └── chroma_task2_10k/                # Task 2: ChromaDB index (29,667 chunks)
├── src/
│   └── task2_chunk_embed_index.py           # Task 2: sampling, chunking, embedding
├── notebooks/
│   └── 01_task1_eda_preprocessing.ipynb     # Task 1: EDA & preprocessing
├── figures/                                 # Task 1 plots
├── requirements.txt                         # All dependencies listed
├── .gitignore                               # Ignores .venv/, vector_store/, pycache, etc.
└── README.md


**Repository Best Practices Met:**
- **Configuration Files** — `.gitignore` excludes sensitive data, virtual environments, large files, vector store binaries, `__pycache__`, etc. `requirements.txt` lists all used packages.
- **README Documentation** — Clear project description, setup instructions, usage, real outputs, and progress.
- **Folder Structure** — Logical organization: notebooks/ (EDA), src/ (scripts), data/ (processed + vector_store), figures/.
- **File Organization** — Clean separation of notebooks, source code, data, and config files.

## Tasks Completed (Implemented by bekonad)

### Task 1 – Exploratory Data Analysis & Preprocessing

**Performed by me:**  
- Loaded first 1,200,000 rows (due to 8 GB RAM limit)  
- Analyzed distribution, narrative presence, length  
- Filtered to target products + non-empty narratives  
- Cleaned text (lowercase, boilerplate removal, special chars, whitespace normalization)  
- Saved → `data/processed/filtered_complaints.csv`

**Real Output from My Run:**

Configuration
Raw file:        ../data/raw/complaints.csv
Output CSV:      ../data/processed/filtered_complaints.csv
Figures folder:  ../figures
Loading rows:    1,200,000
Loading first 1,200,000 rows ...
Shape: (1200000, 18)
Columns: ['Date received', 'Product', 'Sub-product', 'Issue', 'Sub-issue', 'Consumer complaint narrative', ... , 'Complaint ID']
Date range: 2011-12-12 → 2025-06-23


**Top products before filtering (partial):**
- Credit reporting...: 1,066,285
- Debt collection: 57,749
- Credit card: 20,444
- Checking or savings account: 16,070
- Money transfer...: 15,917

**After filtering & cleaning:**
- Shape: (26,168, 18)
- Distribution:
  - Money transfer, virtual currency, or money service: 10,823
  - Checking or savings account: 7,451
  - Credit card: 6,388
  - Payday loan, title loan, personal loan, or advance loan: 942
  - Prepaid card: 517
  - Credit card or prepaid card: 47
- Cleaning: 0 rows removed

**Saved file:** `data/processed/filtered_complaints.csv` (~55.1 MB)

### Task 2 – Text Chunking, Embedding & Vector Store Indexing

**Performed by me:**  
- Stratified sample of 10,000 complaints  
- Chunking: 500 char size, 50 overlap  
- Embedding: `all-MiniLM-L6-v2`  
- Indexed in ChromaDB (batched add)  
- Persisted to `data/vector_store/chroma_task2_10k`

**Real Output from My Successful Run:**

Loading filtered complaints from Task 1...
Columns found: ['Date received', 'Product', 'Sub-product', 'Issue', 'Sub-issue', 'Consumer complaint narrative', ... , 'Complaint ID', 'cleaned_narrative']
Product distribution in full filtered data:
Money transfer, virtual currency, or money service    0.4136
Checking or savings account                           0.2847
Credit card                                           0.2441
Payday loan, title loan, personal loan, or advance loan 0.0360
Prepaid card                                          0.0198
Credit card or prepaid card                           0.0018
Creating stratified sample of 10000 complaints...
Product distribution in sample: (exact match)
Chunking complaint narratives...
→ Created 29,667 chunks from 10000 complaints
Generating embeddings...
Batches: 100%|█████████████████████████████████████████| 464/464 [13:37<00:00, 1.76s/it]
Creating and persisting Chroma collection...
Adding 29,667 chunks in batches of 5000...
Adding batch 0 – 5,000 (5,000 items)
Adding batch 5,000 – 10,000 (5,000 items)
Adding batch 10,000 – 15,000 (5,000 items)
Adding batch 15,000 – 20,000 (5,000 items)
Adding batch 20,000 – 25,000 (5,000 items)
Adding batch 25,000 – 29,667 (4,667 items)
Success!
Vector store saved → data/vector_store/chroma_task2_10k
Collection name: complaints_10k_sample
Total chunks stored: 29667


## Setup Instructions

```bash
git clone https://github.com/bekonad/complaint-chatbot-wk7.git
cd complaint-chatbot-wk7
.venv\Scripts\activate
pip install -r requirements.txt
python src/task2_chunk_embed_index.py

Remaining Tasks

Task 3: RAG pipeline + evaluation (pre-built full store)
Task 4: Interactive chat UI (Gradio/Streamlit)

References

ChromaDB Docs
Sentence Transformers
Hugging Face RAG
Challenge Document: Week 7 PDF


Author: bekonad
Last updated: January 03, 2026