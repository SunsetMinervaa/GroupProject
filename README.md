# NLP in Humanities Group Project

A comprehensive project for analyzing translation strategies in game localization using Qwen3-Embedding models. This project demonstrates semantic similarity analysis, semantic search, clustering visualization, and translation style classification.

## Table of Contents

- [Project Overview](#project-overview)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Scripts Guide](#scripts-guide)
- [Analysis Guide](#analysis-guide)
- [Visualization Guide](#visualization-guide)
- [Translation Style Classifier](#translation-style-classifier)
- [Saving and Recording Results](#saving-and-recording-results)
- [Troubleshooting](#troubleshooting)
- [Changelog](#changelog)
- [Development Notes](#development-notes)

---

## Project Overview

This project analyzes translation strategies (domestication vs. foreignization) in game localization using semantic embeddings. Based on Hollow Knight: Silksong translation data, it compares official Chinese translations with re-translated versions to explore translation strategies.

### Key Features

- **Semantic Similarity Analysis**: Compare official translations vs. re-translations
- **Semantic Search**: Fast semantic search engine with pre-computed embeddings
- **Clustering Visualization**: 2D/3D interactive visualizations of translation strategies
- **Translation Style Classifier**: Classify translations as "domesticated" or "foreignized"

### Tech Stack

- **Model**: Qwen3-Embedding (4B or 8B, 4096-dimensional vectors)
- **Core Libraries**: sentence-transformers, torch, transformers
- **Data Processing**: numpy, scikit-learn
- **Visualization**: plotly (interactive HTML)
- **Dimensionality Reduction**: t-SNE, UMAP (optional)

---

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Optional: Install UMAP for better dimensionality reduction
pip install umap-learn
```

### 2. Model Configuration

Edit `config.py` to select model size:

```python
MODEL_SIZE = "4B"  # Default, recommended for most users (~8GB)
# MODEL_SIZE = "8B"  # For ultimate performance (~16GB)
```

### 3. Run Example Scripts

```bash
# Basic usage: text to embedding
python 00_input_output.py

# Basic usage with examples
python 01_basic_usage.py

# Similarity calculation
python 02_similarity_calculation_2.py

# Semantic search (interactive)
python 03_semantic_search_2.py

# Clustering visualization
python 04_text_clustering_visualization_2.py

# Translation style classifier
python 05_style_classifier.py
```

### 4. First Run

- First run will automatically download the Qwen3-Embedding model (~8GB for 4B, ~16GB for 8B)
- Download time: ~12 minutes (4B) or ~25 minutes (8B) depending on network
- Model will be cached for future use

---

## Project Structure

```
embeddings-main/
├── config.py                          # Model configuration (4B/8B selection)
├── requirements.txt                   # Python dependencies
├── cleaned.json                       # Cleaned translation data (5394 records)
├── style_classifier.joblib            # Trained translation style classifier
│
├── 00_input_output.py                # Basic embedding pipeline
├── 01_basic_usage.py                 # Basic usage examples
├── 02_similarity_calculation_2.py     # Similarity analysis
├── 03_semantic_search_2.py            # Semantic search engine
├── 04_text_clustering_visualization_2.py  # Clustering visualization
├── 05_style_classifier.py            # Translation style classifier
├── load_and_use_classifier.py        # Load pre-trained classifier
│
├── clean_json.py                     # Data cleaning script
├── download_helper.py                 # Model download helper
│
└── Generated files:
    ├── translation_comparison_2d.html
    ├── translation_comparison_3d.html
    └── style_classifier.joblib
```

### Data Structure

`cleaned.json` contains translation triples:

```json
{
  "original_en_text": "English source text",
  "original_zh_text": "Official Chinese translation",
  "translated_text": "Re-translated version"
}
```

---

## Scripts Guide

### 00_input_output.py

**Purpose**: Simplest embedding pipeline - input text, output vector.

**Usage**:
```bash
python 00_input_output.py
```

**Features**:
- Interactive text input
- Real-time embedding generation
- Vector statistics display

### 01_basic_usage.py

**Purpose**: Demonstrates embeddings at different granularities (words, phrases, sentences).

**Usage**:
```bash
python 01_basic_usage.py
```

**Features**:
- Word-level embeddings
- Phrase-level embeddings
- Sentence-level embeddings
- Multilingual comparison
- Statistical analysis

### 02_similarity_calculation_2.py

**Purpose**: Analyze semantic differences between official translations and re-translations.

**Usage**:
```bash
python 02_similarity_calculation_2.py
```

**Default Settings**:
- Sample size: 100 records (random sampling)
- Detailed display: First 10 records
- Device: CPU

**Modify Sample Size**:
Edit line 308:
```python
triples = load_translation_data("cleaned.json", sample_size=100)  # Change this
```

**Output**:
- Semantic similarity analysis
- Strategy difference metrics
- Localization degree analysis
- Statistical summary

### 03_semantic_search_2.py

**Purpose**: Optimized semantic search engine with pre-computed embeddings.

**Usage**:
```bash
python 03_semantic_search_2.py
```

**Features**:
- Pre-computes embeddings for fast search
- Interactive keyword search
- Similarity threshold filtering
- Returns top-N most relevant translation triples

**Note**: Uses first 100 triples by default (modify `max_triples` in code for more).

### 04_text_clustering_visualization_2.py

**Purpose**: Visualize translation strategies in semantic space (2D/3D).

**Usage**:
```bash
python 04_text_clustering_visualization_2.py
```

**Output Files**:
- `translation_comparison_2d.html` - 2D scatter plot
- `translation_comparison_3d.html` - 3D scatter plot (rotatable)

**Default Settings**:
- Sample size: 30 triples (90 text points)
- Perplexity: 30

**Modify Sample Size**:
Edit line 411:
```python
visualizer.load_data("cleaned.json", sample_size=30)  # Change this
```

### 05_style_classifier.py

**Purpose**: Train a classifier to distinguish "domesticated" vs. "foreignized" translations.

**Usage**:
```bash
python 05_style_classifier.py
```

**Features**:
- Trains on 2000 samples (configurable)
- Saves classifier to `style_classifier.joblib`
- Interactive classification mode
- Evaluation metrics (precision, recall, F1)

**Using Pre-trained Classifier**:
```bash
python load_and_use_classifier.py
```

---

## Analysis Guide

### Data Overview

- **Source**: Hollow Knight: Silksong localization files
- **Total Records**: 5394 translation triples
- **Fields**:
  - `original_en_text`: English source (HTML tags removed)
  - `original_zh_text`: Official Chinese translation
  - `translated_text`: Re-translated version

### Analysis Outputs

#### 1. Single Translation Comparison

Each translation pair displays:
- English source text
- Official Chinese translation
- Re-translated version
- Semantic similarity (with visual progress bar)
- Strategy difference
- Localization degree

#### 2. Statistical Summary

- Average semantic similarity
- Average strategy difference
- Average localization degree
- Statistics by text length category

#### 3. Research Conclusions

- Translation strategy tendency (domestication/foreignization)
- Translation consistency analysis
- Localization degree comparison

### Research Directions

1. **Proper Noun Translation**
   - Character names, place names
   - Search for specific terms in `cleaned.json`

2. **Text Type Analysis**
   - Dialogue texts
   - Poetry/verse
   - Narrative texts
   - System prompts

3. **Translation Strategy Patterns**
   - Literary translation vs. literal translation
   - Cultural adaptation vs. source language retention
   - Formal vs. colloquial

---

## Visualization Guide

### Visual Elements

| Type | Color | Symbol | Description |
|------|-------|--------|-------------|
| English Source | 🟢 Green | ● Circle | Source text, reference baseline |
| Domesticated Translation | 🔵 Blue | ■ Square | Official Chinese (original_zh_text) |
| Foreignized Translation | 🔴 Red | ◆ Diamond | Re-translation (translated_text) |

### Interactive Features

#### 2D Plot
- **Hover**: Display text content
- **Click legend**: Hide/show specific types
- **Drag**: Move view
- **Scroll**: Zoom

#### 3D Plot
- **Drag**: Rotate view
- **Scroll**: Zoom
- **Shift + Drag**: Pan
- **Double-click**: Reset view

### Distance Analysis

The script calculates average distances between translation strategies and English source:

```
Distance Analysis: Calculating semantic distances
================================================================================

Average distance (Domesticated -> English): 12.3456
Average distance (Foreignized -> English): 15.7890
Difference (Foreignized - Domesticated): +3.4434

Conclusion: Domesticated translations are closer to the English source.
  -> The domestication strategy may better preserve the core semantics.
```

**Interpretation**:
- **Smaller value** = Translation closer to source semantics
- **Positive difference** = Foreignized distance larger (domesticated closer)
- **Negative difference** = Domesticated distance larger (foreignized closer)

### Parameter Tuning

#### Modify Sample Size

```python
# Quick test (50 triples = 150 points)
visualizer.load_data("cleaned.json", sample_size=50)

# Standard analysis (200 triples = 600 points)
visualizer.load_data("cleaned.json", sample_size=200)

# Large-scale analysis (500 triples = 1500 points, slower)
visualizer.load_data("cleaned.json", sample_size=500)

# All data (5394 triples = 16182 points, very slow)
visualizer.load_data("cleaned.json", sample_size=None)
```

#### Adjust Dimensionality Reduction Parameters

t-SNE `perplexity` parameter:

```python
# Small dataset (<100 triples)
visualizer.reduce_dimensions_2d(perplexity=15)

# Medium dataset (100-300 triples) - default
visualizer.reduce_dimensions_2d(perplexity=30)

# Large dataset (>300 triples)
visualizer.reduce_dimensions_2d(perplexity=50)
```

**Rule**: perplexity typically between 5-50, recommended 1/10 to 1/5 of sample size.

### Observation Points

1. **Clustering Patterns**: Do three text types form separate clusters?
2. **Relative Distance**: Which is closer to green (English) - blue (domesticated) or red (foreignized)?
3. **Distribution Shape**: Tight clusters vs. scattered distribution
4. **Special Cases**: Find outliers (very far or very close points)

---

## Translation Style Classifier

### Training

The classifier is trained to distinguish between:
- **Domesticated Translation** (Label 0): Official Chinese translation
- **Foreignized Translation** (Label 1): Re-translated/literal version

### Usage

**Train new classifier**:
```bash
python 05_style_classifier.py
```

**Use pre-trained classifier**:
```bash
python load_and_use_classifier.py
```

### Performance

With 2000 training samples:
- Accuracy: ~87.5%
- Balanced precision/recall for both classes

### Model Files

- `style_classifier.joblib`: Saved classifier (can be shared)
- Requires: `config.py`, Qwen3-Embedding model, dependencies

---

## Saving and Recording Results

This section provides multiple methods to save and record your analysis results.

### Method 1: Command Line Output Redirection (Simplest)

**Save all console output to a file**:

**Windows (PowerShell)**:
```powershell
python 02_similarity_calculation_2.py > results.txt 2>&1
```

**Windows (CMD)**:
```cmd
python 02_similarity_calculation_2.py > results.txt 2>&1
```

**Linux/Mac**:
```bash
python 02_similarity_calculation_2.py > results.txt 2>&1
```

**With timestamp**:
```bash
python 02_similarity_calculation_2.py > results_$(date +%Y%m%d_%H%M%S).txt 2>&1
```

**View output while saving**:

**Windows (PowerShell)**:
```powershell
python 02_similarity_calculation_2.py | Tee-Object -FilePath results.txt
```

**Linux/Mac**:
```bash
python 02_similarity_calculation_2.py | tee results.txt
```

### Method 2: Using the Results Helper Module

The project includes `save_results_helper.py` with utility functions for saving results.

#### For Similarity Analysis

Add to the end of `main()` function in `02_similarity_calculation_2.py`:

```python
from save_results_helper import save_analysis_results

# At the end of main(), before interactive mode:
save_analysis_results(results, triples, output_file="similarity_results.json")
```

#### For Classification

Add after evaluation in `05_style_classifier.py`:

```python
from save_results_helper import save_classification_report

# After evaluate_classifier():
y_pred = clf.predict(X_test_emb)
save_classification_report(y_test, y_pred, output_file="classification_report.txt")
```

#### For Embeddings

Save embeddings to avoid recomputing:

```python
from save_results_helper import save_embeddings, load_embeddings

# After generating embeddings:
save_embeddings(X_train_emb, X_train_texts, y_train, output_file="train_embeddings.npz")

# Later, load them:
train_data = load_embeddings("train_embeddings.npz")
X_train_emb = train_data["embeddings"]
X_train_texts = train_data["texts"]
y_train = train_data["labels"]
```

### Method 3: Create Timestamped Results Directory

Organize all results in a timestamped folder:

```python
from save_results_helper import create_results_directory
import os

# Create results directory
results_dir = create_results_directory("analysis_results")

# Save all outputs there
save_analysis_results(results, triples, 
                     output_file=os.path.join(results_dir, "similarity_results.json"))
save_classification_report(y_test, y_pred,
                          output_file=os.path.join(results_dir, "classification_report.txt"))

# Copy visualization files
import shutil
shutil.copy("translation_comparison_2d.html", results_dir)
shutil.copy("translation_comparison_3d.html", results_dir)
```

### Method 4: Python Logging Module

For more advanced logging:

```python
import logging
from datetime import datetime

# Setup logging
log_filename = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # Also print to console
    ]
)

# Use in your code
logging.info("Starting analysis...")
logging.info(f"Loaded {len(triples)} translation pairs")
```

### Method 5: Save Specific Metrics to CSV

For statistical analysis in Excel/Python:

```python
import pandas as pd

# Convert results to DataFrame
df = pd.DataFrame(results)

# Add text information
df['english_source'] = [t['original_en_text'] for t in triples]
df['official_translation'] = [t['original_zh_text'] for t in triples]
df['re_translation'] = [t['translated_text'] for t in triples]

# Save to CSV
df.to_csv('analysis_results.csv', index=False, encoding='utf-8-sig')
print("✓ Results saved to analysis_results.csv")
```

### Recommended Workflow

**For Similarity Analysis**:
1. Run with output redirection: `python 02_similarity_calculation_2.py > results.txt 2>&1`
2. Add result saving to script: `save_analysis_results(results, triples, "similarity_results.json")`
3. Results include: Console output (text), structured JSON data, summary statistics

**For Classification**:
1. Run classifier: `python 05_style_classifier.py > classifier_training.txt 2>&1`
2. Results automatically saved: `style_classifier.joblib` (trained model)

**For Visualization**:
1. HTML files automatically saved: `translation_comparison_2d.html`, `translation_comparison_3d.html`
2. Copy to results directory: `mkdir results_$(date +%Y%m%d_%H%M%S) && cp *.html results_*/`

### File Formats Explained

| Format | Best For | Contains | Use Case |
|--------|----------|----------|----------|
| **JSON** | Structured data | All metrics, summary statistics, full texts | Further analysis, data sharing |
| **TXT** | Human-readable | Formatted output, progress messages | Documentation, presentations |
| **CSV** | Excel analysis | Tabular data, one row per sample | Statistical analysis, plotting |
| **NPZ** | Embeddings | Numpy arrays, texts, labels | Avoid recomputing, share vectors |

### Quick Reference

| Task | Command/Code |
|------|-------------|
| Save console output | `python script.py > output.txt 2>&1` |
| Save analysis results | `save_analysis_results(results, triples, "results.json")` |
| Save classification report | `save_classification_report(y_test, y_pred, "report.txt")` |
| Save embeddings | `save_embeddings(embeddings, texts, labels, "embeddings.npz")` |
| Create results dir | `create_results_directory("my_analysis")` |

### Tips

1. **Always use timestamps** in filenames to avoid overwriting
2. **Save both structured (JSON) and human-readable (TXT) formats**
3. **Keep console output** for debugging and progress tracking
4. **Save embeddings** if you'll re-run analysis with different parameters
5. **Create results directories** to keep experiments organized
6. **Document parameters** used in each run (sample_size, batch_size, etc.)

---

## Troubleshooting

### Model Download Issues

#### Problem: Download stuck at "Fetching files: 0%"

**Solution 1: Use Mirror (Recommended for China)**

```bash
# Set environment variable
export HF_ENDPOINT=https://hf-mirror.com

# Or permanently (add to ~/.bashrc or ~/.zshrc)
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.zshrc
source ~/.zshrc
```

**Solution 2: Use ModelScope (Fastest for China)**

The project already uses ModelScope mirror in `config.py`. If download fails, install:

```bash
pip install modelscope
```

**Solution 3: Increase Timeout**

```bash
export HF_HUB_DOWNLOAD_TIMEOUT=600  # 10 minutes
```

**Solution 4: Manual Download**

1. Visit: https://hf-mirror.com/Qwen/Qwen3-Embedding-4B/tree/main
2. Download files to `~/.cache/modelscope/hub/models/Qwen/Qwen3-Embedding-4B/`
3. Re-run script

### Memory Issues

**If memory insufficient**:
- Use 4B model (set in `config.py`)
- Use CPU mode (slower but less memory)
- Reduce batch size
- Reduce sample size

### Performance Issues

**If running too slow**:
- Use GPU if available (change `device="cpu"` to `device="cuda"`)
- Reduce sample size
- Use smaller batch size

### Encoding Errors

- Ensure all files use UTF-8 encoding
- Windows users: Check console encoding settings

### Visualization Issues

**If HTML files don't open properly**:
- Use modern browser (Chrome, Firefox, Edge)
- Check file size (should be ~5MB)
- Check browser console for errors

---

## Changelog

### v2.0.0 - 2025-10-22

#### Major Updates: Multi-Model Support System

**New Features**:
- **Model Configuration System** ⭐
  - New `config.py` for centralized model selection
  - Support for **4B** and **8B** model sizes
  - Default: **4B model** (recommended for most users)
  - Helper functions: `get_model_name()`, `get_model_info()`, `print_model_info()`

**Model Comparison**:

| Model | Parameters | Size | Memory | C-MTEB | MTEB | Recommended |
|-------|-----------|------|--------|--------|------|-------------|
| 4B (default) | 4B | ~8GB | 8-10GB | 72.27 | 69.45 | ✅ Most users |
| 8B | 8B | ~16GB | 16GB+ | 73.84 | 70.58 | Ultimate performance |

**Performance Advantages (4B vs 8B)**:
- Download time: **52% faster** (12 min vs 25 min)
- Model loading: **47% faster** (8s vs 15s)
- Inference speed: **40% faster**
- Memory usage: **50% less** (8-10GB vs 16GB+)
- Performance loss: Only **2%** (C-MTEB: 72.27 vs 73.84)

**Script Updates**:
- All scripts now use `config.py` for model selection
- Display model configuration on startup
- Backward compatible

### v1.1.0 - 2025-10-21

**New Features**:
- Enhanced `01_basic_usage.py` with word/phrase/sentence level demonstrations
- Added comparison examples and statistical analysis
- Added key findings summary

### v1.0.0 - Initial Version

**Initial Features**:
- Four core example scripts
- Qwen3-Embedding-8B model support
- Complete project documentation

---

## Development Notes

### Model Selection

- Default uses **4B model** (98% performance, 8GB memory)
- For best performance, change to `MODEL_SIZE = "8B"` in `config.py`
- After switching, all scripts automatically use the new model

### Memory Management

If memory insufficient:
- Use 4B model (set in config.py)
- Use `model.half()` for half-precision
- Use CPU mode (slower)
- Reduce batch size

### Configuration Import

All scripts need `import config`. Ensure `config.py` is in the same directory.

### Data Saving

Can use `numpy.save()` and `numpy.load()` to save/load embeddings, avoiding redundant computation.

### Interactive Scripts

Scripts with interactive loops (03, 05):
- Use Ctrl+C to interrupt
- Or enter 'quit', 'exit', 'q' to exit

### HTML Visualization

Generated HTML files are large (~5MB). Open with browser for interaction.

### Encoding

All scripts use UTF-8 encoding (`# -*- coding: utf-8 -*-`), ensuring correct handling of multilingual text.

---

## Model Features

**Qwen3-Embedding Series**:

| Feature | 4B Model | 8B Model |
|---------|----------|----------|
| Output Dimension | 4096 | 4096 |
| Supported Languages | 100+ | 100+ |
| MTEB Multilingual | 69.45 | 70.58 (#1) |
| C-MTEB Chinese | 72.27 | 73.84 |
| Model Size | ~8GB | ~16GB |
| Memory Required | 8-10GB | 16GB+ |
| Inference Speed | 1.4x | 1.0x |
| Default Config | ✅ Yes | No |

---

## Application Scenarios

1. **Semantic Search**: Document retrieval, knowledge base Q&A
2. **Text Classification**: Classification based on semantic similarity
3. **Recommendation Systems**: Content recommendations, similar article suggestions
4. **Clustering Analysis**: Topic discovery, content grouping
5. **Deduplication**: Similar content identification
6. **Cross-language Retrieval**: Multilingual document matching
7. **Translation Analysis**: Translation strategy research, localization studies

---

## License

This project is for educational and research purposes.

---

## Contact & Feedback

For questions or suggestions, please refer to the project documentation or create an issue.

**Happy Researching!** 🎉
