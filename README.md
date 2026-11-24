# Self-Correcting RAG

Setup:
    python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    cp .env.example .env    # add your keys
    # put PDFs in data/, then:
    python -m src.ingestion
    streamlit run app.py
