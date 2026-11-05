# JaydariGPT + Jaydari RAG

**JaydariGPT** is a small AI model that we trained ourselves. **Jaydari RAG** is a web API that lets it answer questions about your own code and documents.

```text
   PART 1: jaydariGPT  (Google Colab, GPU)            PART 2: jaydari_rag  (your Mac, this folder)
 ┌──────────────────────────────────────────┐       ┌─────────────────────────────────────────────┐
 │ TinyLlama (1.1B) + 7,000 Alpaca examples │       │ FastAPI server                              │
 │            │ fine-tune with QLoRA        │       │   /ingest-path → read files → Qdrant Cloud  │
 │            ▼                             │ copy  │   /query       → search → TinyLlama+adapter │
 │   jaydari_gpt/ adapter (16 MB) ──────────┼──────►│                  → answer + sources         │
 └──────────────────────────────────────────┘       └─────────────────────────────────────────────┘
```

> **Who is this for?** Anyone joining the project, including juniors. You don't need to know AI beforehand. Read from top to bottom, and check the **Dictionary** whenever you meet a new word.

---

## Table of contents

1. [Dictionary: words you need to know](#1-dictionary-words-you-need-to-know)
2. [The journey: what we built, in exact order](#2-the-journey-what-we-built-in-exact-order)
3. [Part 1: jaydariGPT (training the model)](#3-part-1-jaydarigpt-training-the-model)
4. [Part 2: jaydari_rag (this API)](#4-part-2-jaydari_rag-this-api)
5. [Setup: run it on your Mac](#5-setup-run-it-on-your-mac)
6. [Using the API (Postman / curl)](#6-using-the-api-postman--curl)
7. [Test questions](#7-test-questions)
8. [Run it on a Google Colab GPU (optional)](#8-run-it-on-a-google-colab-gpu-optional)
9. [Troubleshooting: common errors](#9-troubleshooting-common-errors)
10. [Known limitations and next steps](#10-known-limitations-and-next-steps)

---

## 1. Dictionary: words you need to know

| Word | Simple meaning | Example |
|---|---|---|
| **Token** | A small piece of text. Models read tokens, not letters or whole words. | `"headerandshoulder"` → `▁header` `and` `should` `er` |
| **Tokenizer** | The tool that turns text into token numbers (IDs) and back. | `"Hello how are you doing?"` → `[1, 15043, 920, 526, 366, 2599, 29973]` |
| **LLM** | Large Language Model. It writes text by guessing the next token, again and again. | TinyLlama |
| **TinyLlama** | The small open LLM we use (1.1 billion parameters). | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| **Parameter** | One number that the model learned. A model is millions or billions of them. | TinyLlama has 1,100,048,384 |
| **Embedding** | A list of numbers that represents the *meaning* of a text. Similar meanings get similar numbers. | `"hello"` → `[0.41, 0.22, -0.77, ...]` |
| **Vector** | Another name for an embedding (a list of numbers). | 384 numbers per chunk in this project |
| **Vector database** | A database built to store vectors and find the *most similar* ones quickly. | **Qdrant** |
| **Chunk** | A small piece of a file. Big files are cut into chunks before saving. | 1,200 characters, or one Python function |
| **Fine-tuning** | Extra training of an existing model on our own examples. | Teaching TinyLlama to follow instructions |
| **LoRA** | A cheap way to fine-tune: freeze the big model and train only tiny add-on layers. | We trained only 0.1% of the parameters |
| **QLoRA** | LoRA on a model that was shrunk to 4-bit (*quantized*) to save GPU memory. | What our training notebook does |
| **Adapter** | The small file that LoRA training produces. Put it on top of the base model to get the fine-tuned model. | `jaydari_gpt/adapter_model.safetensors` (2.2 MB) |
| **RAG** | Retrieval-Augmented Generation. **Find** relevant pieces of your files first, then **give them to the LLM** with the question. | "Which route creates an item?" → finds `app.js` → answers |
| **API / endpoint** | A URL that a program can call. | `POST /query` |
| **DTO** | Data Transfer Object: the exact shape of the JSON going in or out. | `QueryRequest`, `QueryResponse` |

---

## 2. The journey: what we built, in exact order

```text
 STEP 1  tokenization.ipynb      Learn: text → tokens → IDs → embeddings (by hand, with PyTorch)
   │
 STEP 2  tekoneization.ipynb     Learn: the real TinyLlama tokenizer, padding, attention mask
   │
 STEP 3  jaydari_gpt.ipynb       Fine-tune TinyLlama with QLoRA on Google Colab (T4 GPU)
   │                             → download jaydari_gpt.zip (the adapter)
   │
 STEP 4  jaydari_rag commit 1    FastAPI project + config + GET /health
 STEP 5  jaydari_rag commit 2    Embedding service + Qdrant service + DTOs
 STEP 6  jaydari_rag commit 3    Fix config + add the jaydari_gpt/ adapter folder
 STEP 7  jaydari_rag commit 4    RAG service + POST /ingest-path + POST /query
   │
 STEP 8  Run locally             Install conda, fix startup bugs, ingest a project, ask questions
```

| Step | Where | What happened | Main tech |
|---|---|---|---|
| 1 | `jaydariGPT/tokenization.ipynb` | Split a sentence into words, gave each word an ID, decoded it back, and turned IDs into vectors with `nn.Embedding`. | Python, PyTorch |
| 2 | `jaydariGPT/tekoneization.ipynb` | Loaded TinyLlama's tokenizer (32,000 tokens). Learned special tokens, sub-word tokens, padding and `attention_mask`. | `transformers` |
| 3 | `jaydariGPT/jaydari_gpt.ipynb` | Trained a LoRA adapter on 7,000 instruction examples (2 epochs, 3,500 steps). Tested it and saved it as a zip. | Colab, `bitsandbytes`, `peft`, `trl`, `datasets` |
| 4 | `jaydari_rag` `75a0a48` | Initial configuration and the health-check endpoint. | FastAPI, uvicorn, pydantic-settings |
| 5 | `jaydari_rag` `4f79abb` | Embeddings service, Qdrant service and RAG DTOs. | sentence-transformers, qdrant-client, Pydantic |
| 6 | `jaydari_rag` `d6ba349` | Config fixes, and the trained `jaydari_gpt/` adapter copied into the project. | PEFT |
| 7 | `jaydari_rag` `9285029` | RAG service, ingest and query endpoints. | everything above |
| 8 | your Mac | Installed conda (Miniforge), fixed 3 startup bugs, solved disk-space and slow-ingest problems (see [Troubleshooting](#9-troubleshooting-common-errors)). | conda, Postman |

> The notebooks live in a separate repo: **github.com/asilbek1301/jaydariGPT**. Each one has an **"Open in Colab"** button.

---

## 3. Part 1: jaydariGPT (training the model)

### 3.1 Step 1 and 2: how a model reads text (tokenization)

A model cannot read letters. Everything becomes numbers first:

```text
 "Hello how are you doing?"
          │  tokenizer.tokenize()
          ▼
 ['▁Hello', '▁how', '▁are', '▁you', '▁doing', '?']        ← tokens  (▁ means "a space before")
          │  convert_tokens_to_ids()
          ▼
 [1, 15043, 920, 526, 366, 2599, 29973]                   ← IDs     (1 = <s>, "start of text")
          │  embedding layer
          ▼
 [[0.17, 1.44, ...], [0.65, 0.72, ...], ...]              ← vectors (what the model really uses)
```

**Things we learned:**

- **Vocabulary:** TinyLlama knows 32,000 tokens.
- **Special tokens:** `<s>` start, `</s>` end, `<unk>` unknown. `</s>` is also used as **padding**.
- **Sub-words:** an unknown word is split into known pieces: `headerandshoulder` → `▁header` `and` `should` `er`.
- **Padding and attention mask:** a batch needs every sentence to be the same length. Short sentences are filled with fake tokens (`2`), and `attention_mask` marks them with `0`, so the model ignores them:

```text
 input_ids      [1, 15043, 29892, 920, 526, 366, 29973, 2, 2, 2, 2, 2, 2, 2]
 attention_mask [1,     1,     1,   1,   1,   1,     1, 0, 0, 0, 0, 0, 0, 0]
                 └──────────── real ────────────────┘  └──── padding ────┘
```

### 3.2 Step 3: fine-tuning with QLoRA (`jaydari_gpt.ipynb`)

**Goal:** teach TinyLlama to follow instructions and answer in a clear format.
**Where:** Google Colab with a free **T4 GPU**. Training a model needs a GPU, and a Mac CPU is far too slow.

```text
 ① Load tokenizer           AutoTokenizer("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
          │
 ② Shrink model to 4-bit    BitsAndBytesConfig(load_in_4bit, nf4, double quant, bfloat16 compute)
          │                 "like turning a 4K video into 1080p": much smaller, a little less quality
 ③ Load model               AutoModelForCausalLM(..., quantization_config, device_map="auto")
          │
 ④ Load dataset             yahma/alpaca-cleaned  (51,760 examples) → keep 7,000 → shuffle(seed=42)
          │
 ⑤ Format each example      "### Instruction:\n...\n\n### Input:\n...\n\n### Response:\n..."
          │
 ⑥ Freeze model, add LoRA   LoraConfig(r=8, alpha=16, dropout=0.05, target q_proj + v_proj)
          │                 trainable: 1,126,400 of 1,101,174,784 parameters (0.1023%)
 ⑦ Train                    SFTTrainer, 2 epochs, batch 1 × gradient accumulation 4
          │                 7,000 ÷ 4 = 1,750 steps per epoch → 3,500 steps total
 ⑧ Save + download          model.save_pretrained("jaydari_gpt") → zip → download
          │
 ⑨ Test                     PeftModel(TinyLlama + adapter) → ask_jaydari_gpt("What is Neural Network?")
```

**The training example format** (the model learns where the task ends and the answer starts):

```text
### Instruction:
What are the three primary colors?

### Response:
The three primary colors are red, blue, and yellow. ...
```

When an example has extra context, a `### Input:` section goes between the two.

**Why LoRA and not full fine-tuning?**

```text
 Full fine-tuning                         LoRA / QLoRA (what we did)
 ┌──────────────────────────┐             ┌──────────────────────────┐
 │ update all 1.1B numbers  │             │ 1.1B numbers FROZEN 🔒   │
 │ needs a lot of GPU memory│             │ + tiny lora_A / lora_B   │
 │ → OOM (Out Of Memory) ❌ │             │   layers (1.1M numbers)  │
 └──────────────────────────┘             │ → fits on a free T4 ✅   │
                                          └──────────────────────────┘
```

Inside every one of the 22 TinyLlama blocks, LoRA adds two small layers next to `q_proj` (Query) and `v_proj` (Value):

| Layer | `lora_A` (shrink to 8) | `lora_B` (expand back) | Total |
|---|---:|---:|---:|
| `q_proj` | 2048 × 8 = 16,384 | 8 × 2048 = 16,384 | 32,768 |
| `v_proj` | 2048 × 8 = 16,384 | 8 × 256 = 2,048 | 18,432 |
| **× 22 blocks** | | | **1,126,400** |

**Training settings**

| Setting | Value | Why |
|---|---|---|
| `per_device_train_batch_size` | 1 | one example at a time fits a small GPU |
| `gradient_accumulation_steps` | 4 | update once every 4 examples (acts like batch 4) |
| `num_train_epochs` | 2 | go through the data twice. More passes can **overfit** (memorize) |
| learning rate | 5e-5 → 0 | default, goes down slowly to 0 |
| `bf16` | True | 16-bit math uses less memory |
| `save_strategy` | `"epoch"` | that is why there are `checkpoint-1750` and `checkpoint-3500` |

**Results** (from `jaydari_gpt/checkpoint-3500/trainer_state.json`):

| | Start (step 20) | Middle (step 1,760) | End (step 3,500) |
|---|---:|---:|---:|
| Loss (lower is better) | 1.52 | 1.26 | **1.16** |
| Token accuracy | 61% | 69% | **70%** |

About 2.68 million tokens were seen during training.

**Before vs after:**
- **Before:** for `"Explain what a tokenizer is?"`, the base model only wrote more questions.
- **After:** for `"What is Neural Network?"`, the fine-tuned model gave a clear, on-topic paragraph.
- The training data is **English only**, so the model does not answer Uzbek well. The test function blocks non-English questions with `langdetect`.

### 3.3 What is inside `jaydari_gpt/`?

The zip downloaded from Colab was unzipped into this project:

```text
jaydari_gpt/
├── adapter_model.safetensors   ← ⭐ the trained LoRA weights (2.2 MB). This is "our model".
├── adapter_config.json         ← LoRA settings + base model name (TinyLlama)
├── tokenizer.json              ← tokenizer files (copied from TinyLlama)
├── tokenizer_config.json
├── chat_template.jinja
├── README.md                   ← auto-generated model card (mostly empty)
├── checkpoint-1750/            ← save after epoch 1 (+ optimizer state, can resume training)
└── checkpoint-3500/            ← save after epoch 2 (+ trainer_state.json with the loss log)
```

> TinyLlama itself (2.2 GB) is **not** in this folder. It downloads automatically from Hugging Face the first time, and the adapter is put on top of it.

---

## 4. Part 2: jaydari_rag (this API)

### 4.1 Why RAG?

TinyLlama does not know **your** code. RAG solves that without more training:

```text
 Without RAG:   question ─────────────────────────────► LLM ──► guesses 🤷
 With RAG:      question ──► find the most similar pieces of YOUR files
                                  │
                                  └──► question + those pieces ──► LLM ──► grounded answer + sources ✅
```

### 4.2 Tech stack

| Tech | What it does here | File |
|---|---|---|
| **Python 3.11** + **conda** | language + isolated environment (`rag_env`) | — |
| **FastAPI** | web framework: routes, validation, docs page at `/docs` | `app/main.py`, `app/api/routes.py` |
| **uvicorn** | the server that runs FastAPI (`--reload` restarts on file save) | — |
| **Pydantic** | checks request/response JSON (DTOs) | `app/dto/Rag.py` |
| **pydantic-settings** | reads settings from `.env` | `app/core/config.py` |
| **sentence-transformers** (`all-MiniLM-L6-v2`) | text → 384-number embedding | `app/services/embeddings_service.py` |
| **Qdrant Cloud** + `qdrant-client` | stores and searches embeddings (cosine similarity) | `app/services/qdrant_service.py` |
| **transformers** + **torch** | runs TinyLlama | `app/services/llm_service.py` |
| **PEFT** | puts our `jaydari_gpt` adapter on top of TinyLlama | `app/services/llm_service.py` |
| **Postman** | sends test requests to the API | — |

### 4.3 Folder structure

```text
jaydari_rag/
├── app/
│   ├── main.py                     ← 🚀 starts the app, creates all services once (lifespan)
│   ├── api/
│   │   └── routes.py               ← 🌐 endpoints: /health, /ingest-path, /query
│   ├── core/
│   │   └── config.py               ← ⚙️ Settings class (reads .env)
│   ├── dto/
│   │   └── Rag.py                  ← 📦 request/response shapes
│   └── services/
│       ├── embeddings_service.py   ← 🔢 text → vector (MiniLM)
│       ├── qdrant_service.py       ← 🗄️ save/search vectors in Qdrant
│       ├── llm_service.py          ← 🧠 TinyLlama + jaydari_gpt adapter
│       └── rag_service.py          ← 🧩 the brain: ingest, chunk, decide, prompt, answer
├── utils/
│   └── source_formatting.py        ← ✂️ trims sources and wraps them in ```code fences```
├── jaydari_gpt/                    ← 🎓 our fine-tuned LoRA adapter (from Part 1)
├── .env                            ← 🔐 secrets + settings (NOT in git)
├── requirements.txt                ← 📋 Python packages
└── README.md                       ← 📖 you are here
```

**Layers:** `routes` (HTTP) → `RAGService` (logic) → `EmbeddingService` / `QdrantService` / `LocalLLM` (tools). Each layer only talks to the one below it.

### 4.4 Flow A: server startup

Runs **once** when you start `uvicorn`. It is slow the first time because models download.

```text
 uvicorn app.main:app --reload
          │
          ▼
 lifespan() in app/main.py
          │
 ① Settings()                 read .env  (QDRANT_URL, LLM_MODEL, ...)
          │
 ② EmbeddingService           load all-MiniLM-L6-v2           (~90 MB, first time downloads)
          │
 ③ QdrantService              connect to Qdrant Cloud
          │                   collection missing? → create it (size 384, cosine)
 ④ LocalLLM                   load TinyLlama (~2.2 GB, first time downloads)
          │                   + PeftModel → put jaydari_gpt adapter on top
          │                   + text-generation pipeline (CPU, or GPU if CUDA)
 ⑤ RAGService                 combine ②③④
          │
 ⑥ app.state.rag = rag        routes use it with Depends(get_rag_service)
          │
          ▼
 "Application startup complete."  ✅
```

### 4.5 Flow B: `POST /ingest-path` (save a folder into Qdrant)

```text
 Postman: {"path": "/Users/you/Desktop/some-project"}
          │
          ▼
 IngestPathRequest           path must be at least 5 characters, otherwise 422
          │
          ▼
 RAGService.ingest_folder(path)
          │
          │  for every file in the folder (and sub-folders):
          │    ├─ inside .git / .venv / __pycache__ / node_modules?   → ignore
          │    ├─ not readable as UTF-8 text (image, binary)?         → skipped += 1
          │    ├─ empty file?                                         → skipped += 1
          │    ├─ .py file?  → cut into blocks: imports + one block per top-level def/class
          │    └─ other file → cut into 1,200-character chunks with 200 characters overlap
          │
          │  every chunk gets a stable ID = uuid5(path : chunk number : sha1(text))
          │  (the same chunk always gets the same ID → re-ingesting updates, no duplicates)
          ▼
 QdrantService.upsert_documents(chunks)
          │    ├─ EmbeddingService: every chunk → 384 numbers
          │    └─ upsert to Qdrant: vector + payload {text, metadata: {path, chunk}}
          ▼
 Response: {"ingested": 151, "skipped": 1}      ← ingested = number of CHUNKS, not files
```

**Why overlap?** So a sentence that is cut at a chunk border still appears whole in one of the chunks:

```text
 file:    |──────── chunk 0 (0–1200) ────────|
                                    |──────── chunk 1 (1000–2200) ────────|
                                    └ 200 overlap ┘
```

### 4.6 Flow C: `POST /query` (ask a question)

```text
 Postman: {"question": "Which route in this project creates a new item?"}
          │
          ▼
 QueryRequest                question must be at least 10 characters, otherwise 422
          │
          ▼
 RAGService.answer(question)
          │
 ① embed the question         → 384 numbers
 ② search Qdrant              → top 5 most similar chunks (score 0 … 1, higher = more similar)
 ③ filter                     → keep only score ≥ 0.25 (MIN_SCORE)
 ④ DECIDE: use the chunks?    (see the decision tree below)
 ⑤ build context              → best 2 chunks (MAX_SOURCES), max 4,000 characters, with file paths
 ⑥ build prompt               → "Answer the user question using the context ...
          │                       Context: ...  Question: ...  Answer:"
 ⑦ TinyLlama + adapter        → generates up to LLM_MAX_NEW_TOKENS tokens (temperature 0.2)
 ⑧ clean answer               → cut off anything after an extra "Question:" or "Context:"
          │
          ▼
 format_source()              each source trimmed to 800 chars / 40 lines, wrapped in ```js ...```
          │
          ▼
 Response: {"answer": "...", "sources": [{id, text, score, metadata: {path, chunk}}]}
```

**The decision tree in step ④** (`RAG_MODE=auto`, the default):

```text
                         Does the question contain a project word?
          (project, repo, codebase, file, path, where, implemented, implementation,
           class, function, endpoint, api, controller, service, module, route,  or "/" "\")
                           │                                  │
                          YES                                 NO
                           │                                  │
             best score ≥ 0.35 (MAX_SCORE_STRICT)?     "general question"
                 │                    │                       │
                YES                   NO                      │
                 │                    │                       │
          ✅ use your files     ❌ answer from TinyLlama only, sources = []
```

Other modes: `RAG_MODE=always` always uses the chunks when found, and `RAG_MODE=never` (or `RAG_ENABLED=false`) never searches.

### 4.7 Endpoints

| Method | URL | Body | Response |
|---|---|---|---|
| `GET` | `/health` | — | `{"status": "OK"}` |
| `POST` | `/ingest-path` | `{"path": "/absolute/folder/path"}` | `{"ingested": 151, "skipped": 1}` |
| `POST` | `/query` | `{"question": "at least 10 characters"}` | `{"answer": "...", "sources": [...]}` |
| `GET` | `/docs` | — | FastAPI's automatic, clickable API docs page |

### 4.8 Settings (`.env` → `app/core/config.py`)

Names are not case-sensitive: `QDRANT_URL` in `.env` = `qdrant_url` in Python.

| Setting | Default | Required in `.env`? | Meaning |
|---|---|:---:|---|
| `QDRANT_URL` | — | ✅ | your Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | — | ✅ | Qdrant API key (**secret!**) |
| `QDRANT_COLLECTION` | — | ✅ | collection (table) name, for example `documents` |
| `QDRANT_TIMEOUT` | `10.0` | | seconds to wait for Qdrant |
| `EMBEDDING_MODEL` | — | ✅ | `sentence-transformers/all-MiniLM-L6-v2` |
| `LLM_MODEL` | — | ✅ | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| `LLM_ADAPTER_PATH` | none | | `./jaydari_gpt`. Leave it out to use plain TinyLlama |
| `LLM_DEVICE` | `cpu` | | set `cuda` on an NVIDIA GPU (Colab) |
| `LLM_MAX_NEW_TOKENS` | — | ✅ | max answer length in tokens. Lower = faster (try `128` on a Mac) |
| `LLM_TEMPERATURE` | `0.2` | | creativity: 0 = same answer every time, higher = more random |
| `RAG_ENABLED` | `true` | | `false` = never search files |
| `RAG_MODE` | `auto` | | `auto` / `always` / `never` |
| `TOP_K` | `5` | | how many chunks to get from Qdrant |
| `MIN_SCORE` | `0.25` | | ignore chunks below this similarity |
| `MAX_SCORE_STRICT` | `0.35` | | in `auto` mode, the best chunk must reach this |
| `MAX_SOURCES` | `2` | | max chunks given to the LLM |
| `MAX_CONTEXT_CHARS` | `4000` | | max characters of context in the prompt |
| `CHUNK_SIZE_CHARS` | `1200` | | chunk size for non-Python files |
| `CHUNK_OVERLAP_CHARS` | `200` | | overlap between chunks |
| `LOG_LEVEL` | `INFO` | | log detail |

---

## 5. Setup: run it on your Mac

**You need:** a Mac with [Homebrew](https://brew.sh), about **5 GB of free disk space** (TinyLlama alone is 2.2 GB), a free **Qdrant Cloud** account ([cloud.qdrant.io](https://cloud.qdrant.io)), and Postman.

### Step 1: install conda (one time)

We use **Miniforge**, a small, free conda distribution:

```bash
brew install --cask miniforge
conda init "$(basename "$SHELL")"     # connects conda to your terminal
```

**Close the terminal and open a new one.** Your prompt should now start with `(base)`.

```bash
conda --version                       # check: prints a version number
```

### Step 2: create the environment (one time)

```bash
cd ~/Desktop/DEVEX-AGENTIC-FULLSTACK/jaydari_rag
conda create -n rag_env python=3.11 -y
conda activate rag_env
which python                          # must end with .../envs/rag_env/bin/python
pip install -r requirements.txt       # torch is big, so this takes a few minutes
```

In VS Code, press `Cmd+Shift+P` → **Python: Select Interpreter** → choose `rag_env`.

### Step 3: create the `.env` file (one time)

Create `jaydari_rag/.env`. You get the URL and API key from your Qdrant Cloud cluster page.

```dotenv
# Qdrant (Vector Database)
QDRANT_URL=https://<your-cluster-id>.<region>.gcp.cloud.qdrant.io:6333
QDRANT_API_KEY=<your-qdrant-api-key>
QDRANT_COLLECTION=documents

EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
LLM_MAX_NEW_TOKENS=512
LLM_ADAPTER_PATH=./jaydari_gpt
```

> 🔐 **Never commit `.env`.** It is already in `.gitignore`. Never paste your API key in chats or screenshots.

### Step 4: get the adapter

Make sure the `jaydari_gpt/` folder exists with `adapter_model.safetensors` and `adapter_config.json` inside. If it is missing, run `jaydari_gpt.ipynb` on Colab (see [Part 1](#32-step-3-fine-tuning-with-qlora-jaydari_gptipynb)), download `jaydari_gpt.zip`, and unzip it here.

### Step 5: start the server (every time)

```bash
cd ~/Desktop/DEVEX-AGENTIC-FULLSTACK/jaydari_rag
conda activate rag_env
uvicorn app.main:app --reload
```

The **first** start downloads MiniLM (~90 MB) and TinyLlama (~2.2 GB), so give it a few minutes. You are ready when you see:

```text
INFO:     *** JAYDARI RAG STARTS ***
...
INFO:     Application startup complete.
```

Stop the server with `Ctrl+C`.

### Step 6: check it works

Open **http://127.0.0.1:8000/health** in your browser. You should see `{"status":"OK"}`.
Open **http://127.0.0.1:8000/docs** to see and try every endpoint.

---

## 6. Using the API (Postman / curl)

Always in this order: **health → ingest → query.**

**1. Health**

```bash
curl http://127.0.0.1:8000/health
```

**2. Ingest a folder** (Postman: `POST`, Body → raw → JSON)

```bash
curl -X POST http://127.0.0.1:8000/ingest-path \
  -H "Content-Type: application/json" \
  -d '{"path": "/Users/asilbek/Desktop/DEVEX-AGENTIC-FULLSTACK/reja"}'
# → {"ingested": 151, "skipped": 1}
```

> ⚠️ **Choose a small folder.** The code reads *every* text file, including build output like `.next/` and `dist/`. A 1.8 GB project can take a very long time. Point at the source folder (for example `.../my-app/src`) instead.

**3. Ask a question**

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Which route in this project creates a new item?"}'
```

> ⏱️ On a Mac, TinyLlama runs on the **CPU**, so one answer can take **1–2 minutes**. Send **one request at a time**. A running ingest makes queries even slower. To speed it up, lower `LLM_MAX_NEW_TOKENS` (for example `128`), or use [Colab](#8-run-it-on-a-google-colab-gpu-optional).

---

## 7. Test questions

**A. Project questions** (contain a project word, so they should return `sources`):

| Question | Expected source |
|---|---|
| "Which route in this project creates a new item?" | `app.js` → `/create-item` |
| "Which endpoint deletes all plans in the project?" | `app.js` → `/delete-all` |
| "How is the edit-item route implemented?" | `app.js` → `findOneAndUpdate` |
| "Which Node version does the project Dockerfile use?" | `Dockerfile` |
| "What does the README file say about NVM commands?" | `README.md` |

Check two things: does `sources` show the right file, and does the answer agree with it?

**B. General questions** (no project word, so `sources` should be `[]`):
- "What is array in JS"
- "What is a closure in JavaScript?"
- "Explain the difference between let and const"

**C. Edge cases**
- **Not in the project:** "Where is the payment service implemented in this project?" A good answer says it can't find it. Watch whether TinyLlama makes something up.
- **Word trap:** "What is a JavaScript class?" is a general question, but it contains *class*, so the app searches your files anyway.
- **Too short:** "What?" → `422` error (minimum 10 characters).

---

## 8. Run it on a Google Colab GPU (optional)

On a Colab **T4 GPU**, answers take seconds instead of minutes. No code change is needed. The app already uses the GPU when `LLM_DEVICE=cuda`.

1. Upload the whole `jaydari_rag` folder (with `.env` and `jaydari_gpt/`) and any folder you want to ingest to **Google Drive** → `MyDrive`.
2. In Colab: **Runtime → Change runtime type → T4 GPU**.
3. Run these cells:

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!cp -r /content/drive/MyDrive/jaydari_rag /content/
%cd /content/jaydari_rag
!pip install -q -r requirements.txt
!grep -q '^LLM_DEVICE=' .env || echo 'LLM_DEVICE=cuda' >> .env
!python -c "import torch; print('GPU:', torch.cuda.is_available())"
```

```python
!nohup uvicorn app.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
```

```python
# public URL for Postman (Cloudflare tunnel, no account needed)
!wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared && chmod +x cloudflared
!nohup ./cloudflared tunnel --url http://localhost:8000 > tunnel.log 2>&1 &
!sleep 8 && grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' tunnel.log
```

4. In Postman, replace `http://127.0.0.1:8000` with the printed `https://….trycloudflare.com` URL.
   Ingest paths must be Colab paths, for example `{"path": "/content/drive/MyDrive/reja"}`.

> ⚠️ Anyone with the tunnel URL can use your API and your Qdrant. Don't share it, and stop the runtime when you finish. Colab disconnects after ~90 minutes idle, and you get a new URL each session.

---

## 9. Troubleshooting: common errors

Most of these came up while we set the project up for the first time.

| Error / symptom | Why it happens | Fix |
|---|---|---|
| `conda: command not found` | conda is not installed | `brew install --cask miniforge` |
| `CondaError: Run 'conda init' before 'conda activate'` | conda is not connected to your terminal yet | `conda init "$(basename "$SHELL")"`, then **open a new terminal** |
| `ModuleNotFoundError: No module named 'services'` | import was missing the `app.` prefix | use `from app.services.embeddings_service import ...` |
| `AttributeError: embedding_model` | `settings = Settings` (the class), not `Settings()` (an object), so `.env` was never read | write `Settings()` with parentheses |
| Pydantic error: field required (for example `qdrant_url`) | a required value is missing in `.env` | add it (see [Settings](#48-settings-env--appcoreconfigpy)) |
| `Not enough free disk space to download the file` | TinyLlama needs 2.2 GB | free up space (Trash, `~/Library/Caches`, `conda clean --all -y`) and restart |
| Postman stuck on **"Sending request…"** for `/ingest-path` | the folder is huge (for example 1.8 GB with `.next/`, `dist/`) | cancel, restart the server, ingest a smaller folder |
| `/query` takes minutes | TinyLlama runs on the CPU, up to 512 tokens. Another request may be running too | wait 1–2 min, one request at a time, lower `LLM_MAX_NEW_TOKENS`, or use Colab |
| `ingested` looks bigger than your file count | it counts **chunks**, not files | normal |
| Commands hang reading files on the Desktop | the files are **iCloud cloud-only** (Finder shows a ☁ icon) | Finder → right-click the folder → **Download Now** |
| `Address already in use` (port 8000) | an old server is still running | stop it with `Ctrl+C`, or use `--port 8001` |

---

## 10. Known limitations and next steps

These are honest notes for whoever improves the project next.

1. **Prompt format mismatch.** The adapter was trained on `### Instruction:` / `### Response:`, but `rag_service._build_prompt()` uses `Context: ... Question: ... Answer:`. Using the training format would likely give better answers.
2. **Mac GPU is not used.** `llm_service.py` only checks for CUDA (NVIDIA). Apple's `mps` GPU could make local answers much faster.
3. **Build folders are ingested.** Only `.git`, `.venv`, `__pycache__` and `node_modules` are ignored. `.next`, `dist` and `build` are read too, which makes ingest very slow.
4. **Ingest is all-or-nothing.** All chunks are embedded and uploaded in one big batch. A huge folder can time out with nothing saved. Uploading in batches of, say, 256 would be safer.
5. **Keyword routing is simple.** Words like *class* or *function* in a general question send it to file search.
6. **English only.** The training data is English, so Uzbek questions get poor answers.
7. **Training data choice.** In the notebook, `select(range(7000))` runs **before** `shuffle`, so only the first 7,000 of 51,760 examples can ever be used. Shuffle first, then select.
8. **Secrets in ingested code.** Ingest sends file contents to Qdrant Cloud, and answers can show them. Never ingest files that contain passwords or keys. Move secrets into environment variables first.
