## Semantic Chat

Ten dokument opisuje: **trening modelu w `experiments/`**, **przeniesienie wag do platformy**, **konfigurację kluczy API** oraz **uruchomienie całego stacku**.

FastAPI serwis obsługujący chat (FAQ + RAG), embeddingi z modelu `custom_hybrid`, Supabase (baza + storage) oraz OpenAI (generacja odpowiedzi RAG).

---

## Wymagania

| Komponent | Wersja / uwagi |
|-----------|----------------|
| Python | 3.11+ (backend); 3.12 OK dla `experiments/` |
| Node.js | 18+ (frontend) |
| CUDA | opcjonalnie — przyspiesza trening w `experiments/` |
| Supabase | projekt z SQL migrations + bucket `documents` |
| OpenAI | klucz API (RAG / odpowiedzi z dokumentów) |

---

## Klucze API i konfiguracja

### Supabase

1. Utwórz projekt na [supabase.com](https://supabase.com).
2. **Project Settings → API**:
   - **Project URL** → `SUPABASE_URL` (backend) i `VITE_SUPABASE_URL` (frontend)
   - **anon public** → `VITE_SUPABASE_ANON_KEY` (frontend, logowanie użytkownika)
   - **service_role** (secret) → `SUPABASE_KEY` (backend — pełny dostęp do DB/storage; **nie** w frontendzie)
3. **SQL Editor** — uruchom migracje w kolejności:
   - `migration/001_init_tables.sql`
   - `migration/002_conversations.sql`
   - `migration/003_chat_response_source.sql`
4. **Storage** — bucket `documents` (domyślna nazwa; backend tworzy go przy pierwszym uploadzie, jeśli ma uprawnienia).

### OpenAI

1. Konto na [platform.openai.com](https://platform.openai.com).
2. **API keys** → utwórz klucz → `OPENAI_API_KEY` w backend `.env`.
3. Model używany w kodzie: `gpt-4.1-mini` (`app/services/openai_service.py`).

### Plik `.env` (backend)

Utwórz `platform/backend/.env`:

```env
SUPABASE_URL=https://xxxx.supabase.co
SUPABASE_KEY=eyJ...service_role...
OPENAI_API_KEY=sk-...

# Opcjonalne
FRONTEND_ORIGIN=http://localhost:5173
SUPABASE_STORAGE_BUCKET=documents
EMBEDDING_BACKEND=custom
EMBEDDING_RUNTIME=numpy
SEMANTIC_MODEL_PATH=app/models/custom_hybrid_encoder.npz
```

### Plik `.env` (frontend)

Utwórz `platform/frontend/.env`:

```env
VITE_SUPABASE_URL=https://xxxx.supabase.co
VITE_SUPABASE_ANON_KEY=eyJ...anon...
VITE_API_BASE_URL=http://localhost:8000
```

---

## 1. Trening modelu (`experiments/`)

Katalog `experiments/` to osobny projekt badawczy (STS Benchmark). Trenuje architekturę **CompactSimilarityModel** (`custom_hybrid`) i zapisuje checkpointy.

### Instalacja

```bash
cd experiments
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Przy pierwszym uruchomieniu pobierany jest tokenizer `bert-base-uncased` (cache: `data/hf_cache/`).

Opcjonalnie dla baseline'u GloVe: pobierz `glove.6B.100d.txt` do `experiments/data/`.

### Trenowanie

```bash
cd experiments
source venv/bin/activate

# Tylko custom_hybrid (encoder używany w platformie)
python main.py train --models custom

# Pełny pipeline: trening + benchmark wszystkich modeli
python main.py

# Benchmark bez treningu (wymaga istniejących checkpointów)
python main.py benchmark --models custom
```

Checkpoint po treningu:

```text
experiments/results/custom_hybrid/model.pt
```

Metryki i wykresy trafiają do `experiments/results/` (CSV, JSON, PNG) — katalog jest w `.gitignore`.

### Dataset treningowy

- **Źródło:** `mteb/stsbenchmark-sts` (HuggingFace)
- **Format:** pary zdań + similarity score 0–5 (normalizowany do 0–1)
- **Tokenizacja:** WordPiece (`bert-base-uncased`), zamrożone embeddingi BERT 768-d
- **Loss:** HybridLoss (Pearson + Spearman + contrastive + CoSENT)


---

## 2. Przeniesienie modelu do platformy

Produkcja **nie używa PyTorch** — serwis ładuje wagę w formacie `.npz` (NumPy) lub `.onnx`.


Z venv `experiments` (musi mieć `torch` + `transformers`):

```bash
cd platform/backend

# Skopiuj checkpoint z experiments (lub wskaż ścieżkę)
cp ../../experiments/results/custom_hybrid/model.pt app/models/custom_hybrid.pt

# Eksport: encoder .npz + .onnx + tabela tokenów BERT
../../experiments/venv/bin/python scripts/export_custom_encoder.py \
  --ckpt app/models/custom_hybrid.pt \
  --out app/models/custom_hybrid_encoder.npz \
  --onnx-out app/models/custom_hybrid_encoder.onnx \
  --bert-out app/models/bert_token_embeddings.npz
```

Powstają pliki:

| Plik | Rola |
|------|------|
| `app/models/custom_hybrid_encoder.npz` | Encoder 128-d (domyślny runtime NumPy) |
| `app/models/custom_hybrid_encoder.onnx` | Opcjonalnie: `EMBEDDING_RUNTIME=onnx` |
| `app/models/bert_token_embeddings.npz` | Zamrożona macierz tokenów BERT (~90 MB) |


## 3. Uruchomienie backendu

```bash
cd platform/backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Przed startem upewnij się, że istnieją:

- `app/models/custom_hybrid_encoder.npz`
- `app/models/bert_token_embeddings.npz`

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Sprawdzenie:

- `GET http://localhost:8000/health` → `{"status":"ok"}`
- `GET http://localhost:8000/metrics` → backend encodera + statystyki cache

---

## 4. Uruchomienie frontendu

```bash
cd platform/frontend
npm install
npm run dev
```

Aplikacja: `http://localhost:5173` (Vite może użyć `:5174` — backend akceptuje oba porty w CORS).

---

## Zmienne środowiskowe (backend)

| Zmienna | Wymagane | Domyślnie | Opis |
|---------|----------|-----------|------|
| `SUPABASE_URL` | tak | — | URL projektu Supabase |
| `SUPABASE_KEY` | tak | — | service_role key |
| `OPENAI_API_KEY` | tak | — | Klucz OpenAI (RAG) |
| `FRONTEND_ORIGIN` | nie | localhost:5173 | CORS (można lista po przecinku) |
| `EMBEDDING_BACKEND` | nie | `custom` | `custom` lub `glove` (legacy) |
| `EMBEDDING_RUNTIME` | nie | `numpy` | `numpy` lub `onnx` |
| `SEMANTIC_MODEL_PATH` | nie | `app/models/custom_hybrid_encoder.npz` | Ścieżka do wag encodera |

---

