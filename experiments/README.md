## STS Benchmark Experimental Framework

This project provides a modular, research-grade framework for evaluating multiple neural and baseline architectures on the STS Benchmark dataset using PyTorch.

### Setup

- Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

- (Optional) Download GloVe-100d to `data/glove.6B.100d.txt` if you want to use the GloVe baseline.

### Running Experiments

```bash
python -m project.main
```

Results (metrics, CSV, JSON, and plots) are written to the `results/` directory.

