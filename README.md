# syllabus-rag 📚

A RAG (Retrieval-Augmented Generation) system that lets you upload any course
syllabus and ask questions about it in plain English.

Ask it things like:
- *"What is the grading policy?"*
- *"When are office hours?"*
- *"What are the late submission penalties?"*
- *"What textbooks do I need?"*
- *"What topics does this course cover?"*

Every answer comes word-for-word from your document. Zero hallucination.

---

## Background

I was taking **Big Data Technologies** at Illinois Tech and kept ctrl+F-ing
through the syllabus every time I forgot a deadline or needed to check a policy.
So I built this over a weekend to fix that.

Originally I used TinyLlama as the answer generator — but it kept hallucinating,
inventing policies and offices that didn't exist in the syllabus at all. The fix
ended up being to remove the LLM entirely. Instead of asking a model to summarize
retrieved chunks, I use embedding similarity scores to find the most relevant
chunks and return them directly from the document.

Turns out sometimes the best AI solution is knowing when not to use AI.

---

## How it works

```
Your Question
     │
     ▼
[1] Embed the question into a 384-dim vector
     │
     ▼
[2] FAISS searches for the most similar chunks in your syllabus
     │
     ▼
[3] Most relevant chunks are returned directly as the answer
     │
     ▼
Answer — straight from your document, no model in the loop
```

No LLM is used for generation. Every word in the answer came from your file.

---

## Stack

| Tool | What it does |
|---|---|
| `sentence-transformers` | Embeds text into vectors (all-MiniLM-L6-v2 model) |
| `faiss-cpu` | Stores and searches vectors by similarity |
| `langchain-text-splitters` | Smart chunking that doesn't cut sentences in half |
| `python-docx` | Reads .docx files including tables (grade breakdowns, schedules, etc.) |
| `flask` + `flask-cors` | Backend server that serves the web UI and handles uploads |

Everything runs locally. No API key needed, no data sent anywhere.

---

## Setup

**Clone and install:**

```bash
git clone https://github.com/yourusername/syllabus-rag
cd syllabus-rag
pip install -r requirements.txt
```

**Start the server:**

```bash
python app.py
```

**Open in browser:**

```
http://localhost:5000
```

That's it. Drag and drop your syllabus (.docx or .txt) into the upload zone and
start asking questions.

---

## Project structure

```
syllabus-rag/
├── app.py              # Flask backend — document loading, chunking, embedding, retrieval
├── big-data-rag.html   # Frontend — served by Flask at localhost:5000
├── requirements.txt    # pip dependencies
├── README.md           # you're reading it
└── uploads/            # auto-created, stores uploaded syllabi
```

---

## What I learned building this

**Word tables are invisible to most extractors.**
The grade breakdown (Short Quizzes 14%, Half Term Exam 35%, etc.) lived in a
Word table, not a paragraph. My first version completely missed it because I was
only reading `doc.paragraphs`. Had to also iterate `doc.tables` to get the full
picture. The document went from 6,000 to 12,800 characters once I fixed this.

**Chunk overlap is underrated.**
Without overlap, answers that fall right at a chunk boundary just disappear.
Setting `chunk_overlap=50` means each chunk shares a bit of text with its
neighbors — so context is never lost at the edges.

**LLMs hallucinate on factual Q&A.**
TinyLlama kept inventing details — "Office of Disability Services" that doesn't
exist, grade policies with made-up exemptions, etc. No amount of prompt
engineering fully fixed it. Removing the LLM and returning chunks directly was
the right call for this use case.

**Embedding similarity is a surprisingly good answer signal.**
The L2 distance from FAISS tells you exactly how relevant each chunk is to the
question. If the best score is too high (too far = not relevant), you can
confidently say "I don't have that information" instead of guessing.

---

## Limitations / things I'd improve

- **No conversational memory.** Each question is independent — it doesn't
  remember what you asked before. Adding chat history would make it feel more
  natural.

- **Answer quality depends on chunk quality.** If the relevant information is
  split across multiple chunks, the answer might be incomplete. Tuning
  `chunk_size` and `chunk_overlap` per document type would help.

- **Single document at a time.** Right now uploading a new syllabus replaces
  the old one. Supporting multiple documents simultaneously would be a useful
  extension.

- **No persistent index.** The FAISS index is rebuilt from scratch every time
  you upload. For large documents, saving the index to disk with
  `faiss.write_index()` and reloading it would save time on restart.

---

## Requirements

```
sentence-transformers
faiss-cpu
langchain-text-splitters
python-docx
numpy
flask
flask-cors
```

Install with: `pip install -r requirements.txt`

---

## Resources that helped

- [sentence-transformers docs](https://www.sbert.net/)
- [FAISS getting started](https://faiss.ai/)
- [LangChain text splitter docs](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- Big Data Technologies at Illinois Institute of Technology — shoutout Prof. Rosen

---

*Built as a learning project while studying Big Data Technologies at Illinois Institute of Technology.*
