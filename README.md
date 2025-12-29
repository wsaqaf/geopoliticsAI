Here is the updated **README.md** file, now including the precise **Environment Setup** section with instructions for installing Ollama and the specific 4-bit quantized models as requested.

---

# Book Chapter on Geopolitics & LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Ollama](https://img.shields.io/badge/Ollama-v0.11.10-orange.svg)](https://ollama.com/)

**Official Code & Data Repository**
*Book Chapter: "The Palestine-Israel Conflict in AI Code: How Regional Fine-Tuning Writes Geopolitical Allegiance into LLMs"*
*(Forthcoming in: Geopolitics and Media: Theory of Post-Globalizing Communication)*

---

## 📖 Overview

This repository contains the dataset, prompt instruments, and analysis scripts for the study **"The Algorithmic AI Border."**

This research interrogates the "universal" pretensions of Artificial Intelligence by subjecting two architecturally related Large Language Models (LLMs) to a geopolitical stress-test along the Israel-Palestine fault line. We compare **Gemma-2 (9B)** (the "Hegemonic/Western" baseline) and **Fanar-1 (9B)** (the "Regional/Counter-Hegemonic" fine-tune) to measure the extent of **Algorithmic Borderization**—the phenomenon where AI models reproduce the geopolitical polarization of their training data.

---

## 📂 Repository Structure

```bash
├── data/
│   ├── prompts-geo-test.csv       # The 16 paired counter-narrative prompts (Input)
│   └── benchmark_results.csv      # The full dataset of 640 Likert observations (Output)
├── analysis/
│   └── polarization_analysis.py   # Script to calculate Gaps and T-Tests
├── paper/
│   └── The_Algorithmic_AI_Border_Draft.docx  # Full manuscript draft
├── figures/
│   └── genetic_architecture.png   # Methodology diagram
└── README.md
```

---

## 🛠️ Prerequisites & Setup

This study utilizes **local inference** to ensure reproducibility and avoid the "black box" unpredictability of APIs.

### 1. Install Ollama

Download and install the **Ollama** inference engine. This study was conducted using **Version 0.11.10**.

* **Download:** [https://ollama.com/download](https://ollama.com/download)
* **Verify Installation:**
```bash
ollama --version

```



### 2. Install Quantized Models (4-bit)

Both models must be run in 4-bit quantization to match the study's parameters.

#### **A. Gemma-2 (9B)**

The base model is available directly from the Ollama library (which uses 4-bit quantization by default).

```bash
ollama pull gemma2:9b

```

*Source: [google/gemma-2-9b*](https://huggingface.co/google/gemma-2-9b)

#### **B. Fanar-1 (9B)**

The regional model must be installed from the QCRI Hugging Face repository. You will need to download the 4-bit quantized GGUF file and create a custom model in Ollama.

1. **Download the GGUF:**
Ensure you download the `Q4_K_M.gguf` (4-bit quantized) version from the repository.
*Repo: [QCRI/Fanar-1-9B*](https://huggingface.co/QCRI/Fanar-1-9B)
2. **Create Modelfile:**
Create a file named `Modelfile` in your directory with the following content (update the path to where you saved the GGUF):
```dockerfile
FROM ./fanar-1-9b-Q4_K_M.gguf
PARAMETER temperature 0.7
SYSTEM "You are a helpful AI assistant."

```


3. **Build the Model:**
Run the following command to register Fanar-1 in Ollama:
```bash
ollama create fanar1 -f Modelfile

```



---

## 🔬 Methodology

### 1. The "Genetic" Control

To isolate the variable of **Regional Alignment**, we selected two models that share a direct architectural lineage:

* **Parent Model:** `gemma2:9b` (Western Corpus)
* **Offspring Model:** `fanar1` (Arabic Fine-Tuning + RLHF)

### 2. The Instrument: Zones of Borderization

We designed 16 "Paired Counter-Narrative" prompts across four thematic zones. Each topic was presented with two opposing frames: an **Israeli-Centric Frame** (Security/Statehood) and a **Palestinian-Centric Frame** (Rights/Nakba).

| Zone | Theme | Key Topics |
| --- | --- | --- |
| **Zone A** | **Foundational Legitimacy** | 1948 Displacement, 1947 Partition |
| **Zone B** | **Security** | Hamas, Gaza Blockade |
| **Zone C** | **Sovereign Legality** | East Jerusalem, Settlements |
| **Zone D** | **Systemic Control** | Home Demolitions, Checkpoints |

### 3. Sampling

* **N = 640** total observations.
* **20 Independent Trials** per prompt.
* **Temperature:** 0.7 (Stochastic Control).

---

## 🚀 Reproduction

To replicate the analysis provided in the paper:

1. **Install Python Dependencies:**
```bash
pip install pandas scipy numpy matplotlib

```


2. **Run the Analysis Script:**
This script reads the raw CSV, calculates the Polarization Gap (), and performs the T-Test.
```bash
python analysis/polarization_analysis.py

```


3. **Verify Results:**
The script will output the mean polarization scores per Zone and the statistical significance values reported in the "Empirical Case" section.

---

## Dashboard

You can access the dashboard with simple charts and filters [HERE](https://al-saqaf.se/geobook/).

---
## 📝 Citation

If you use this dataset or code in your research, please cite the book chapter:

```bibtex
@incollection{WalidAlsaqaf2025AlgorithmicBorder,
  author    = {Walid Al-Saqaf},
  title     = {The Algorithmic AI Border: Large Language Models as Geopoliticized Media Spheres},
  booktitle = {Geopolitics and Media: Theory of Post-Globalizing Communication},
  year      = {2025}
}

```

---

## 📧 Contact

For questions regarding the methodology or dataset:

* **Author:** Walid Al-Saqaf
* **Affiliation:** Doha Institute for Graduate Studies
* **Email:** walid.alsaqaf[at]dohainstitute.edu.qa

```

```
