# ⚡ Mini SLM – Van de Graaff Generator Chatbot  

This is a **Mini SLM (Self-Learning Module)** project built with **Streamlit**.  
It is designed to explain concepts about the **Van de Graaff generator** in a simple, interactive way.  

The chatbot uses:  
- **PDF-based context retrieval** (you can swap in *any* PDF you like!)  
- **FAISS vector search** for semantic search  
- **SentenceTransformers** for embeddings  
- **Groq LLM** for generating accurate and concise answers  

You can **adjust the code**, **change the file path**, and **customize it for any topic** – this is a flexible template for building small learning modules (Mini SLMs).  

------------------------------------------------------------
🚀 QUICK SETUP  
------------------------------------------------------------

1️⃣ Clone the Repository  
-----------------------
git clone https://github.com/your-username/mini-slm-van-de-graaff.git
cd mini-slm-van-de-graaff

2️⃣ Install Requirements  
------------------------
Make sure you have Python 3.9+ installed, then run:

pip install -r requirements.txt

3️⃣ Add Your PDF  
----------------
This project comes with a Van de Graaff generator PDF (expected path: docs/VanDeGraaff.pdf).  
But you can replace it with any other PDF — just put your file in docs/ and update PDF_PATH accordingly.

4️⃣ Set Environment Variables  
-----------------------------
You must set your Groq API key before running.

Linux / macOS (bash):
export GROQ_API_KEY="your_api_key_here"
export PDF_PATH="docs/VanDeGraaff.pdf"

Windows (PowerShell):
$env:GROQ_API_KEY="your_api_key_here"
$env:PDF_PATH="docs/VanDeGraaff.pdf"

5️⃣ Run the App  
---------------
streamlit run app.py

Then open the link shown in your terminal (usually http://localhost:8501).

------------------------------------------------------------
📂 PROJECT STRUCTURE  
------------------------------------------------------------

mini-slm-van-de-graaff/
├── app.py              # Streamlit app (you can modify this as needed)
├── requirements.txt    # Python dependencies
├── .gitignore          # Ignore cache/env files
├── README.md           # Project documentation
└── docs/
    └── VanDeGraaff.pdf # Default reference PDF (replaceable)

------------------------------------------------------------
✨ FEATURES  
------------------------------------------------------------

✅ Mini SLM Template – Replace the PDF & teach any topic  
✅ Retrieval-Augmented Generation (RAG) for accurate answers  
✅ FAISS-powered search for fast and relevant context  
✅ LaTeX support – perfect for equations and formulas  
✅ Clean Streamlit interface for a smooth user experience  

------------------------------------------------------------
🛠️ CUSTOMIZATION  
------------------------------------------------------------

- Change PDF_PATH in .env or as an environment variable to use a different file.  
- Adjust chunk_text() or retrieve_top_k() in app.py if you want different chunk sizes or more retrieved context.  
- Swap out model_name='all-MiniLM-L6-v2' for another embedding model if you want better accuracy.  

This makes it a ready-to-use template for creating your own small learning modules.  

------------------------------------------------------------
🛠️ TROUBLESHOOTING  
------------------------------------------------------------

- PDF Not Found: Check that your file exists and matches the PDF_PATH.  
- API Key Missing: Ensure you have set GROQ_API_KEY.  
- Slow Embedding: Use a smaller embedding model for faster processing.  

------------------------------------------------------------
📜 LICENSE  
------------------------------------------------------------

This project is open-source — fork it, modify it, and build your own Mini SLM projects!
