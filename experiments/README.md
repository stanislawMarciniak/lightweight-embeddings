## STS Benchmark Experimental Framework

This project provides a modular, research-grade framework for evaluating multiple neural and baseline architectures on the STS Benchmark dataset using PyTorch.

### Setup

- Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

- Tokenization uses the `bert-base-uncased` fast (WordPiece) tokenizer and 768-d token embeddings from BERT (downloaded automatically on first run, cached under `data/hf_cache`).
- Download GloVe-100d to `data/glove.6B.100d.txt` for the `glove_mean` baseline.
- Optional: place a PCA-compressed BERT matrix at `data/model2vec_static_256.pt` for offline `model2vec_static` fallback (~0.59 Pearson). By default the baseline loads `minishlab/potion-base-8M` via the `model2vec` package (~0.77 Pearson).
- `minilm` downloads `sentence-transformers/all-MiniLM-L6-v2` on first benchmark run.

### Running Experiments

```bash
python -m project.main
```

Results (metrics, CSV, JSON, and plots) are written to the `results/` directory.

