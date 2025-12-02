#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Translation style classifier: determine whether a Chinese sentence
is more like a "domesticated" translation or a "foreignized" translation.

Data source: translation triples in cleaned.json
  - original_zh_text -> domesticated translation (official)
  - translated_text  -> foreignized translation (re-translation / literal)

Pipeline:
  1. Read cleaned.json and build a (text, label) dataset
  2. Use Qwen3-Embedding to encode texts into vectors
  3. Train a simple logistic regression binary classifier on embeddings
  4. Evaluate on a held-out test set and then enter an interactive mode
     to classify user-provided Chinese sentences
"""

import json
import os
import sys
from datetime import datetime
from typing import List, Tuple

import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import config

# import torch  # GPU code temporarily disabled


def load_translation_pairs(
    json_file: str = "cleaned.json",
) -> Tuple[List[str], List[int]]:
    """
    Build Chinese translation samples and labels from cleaned.json.

    Label convention:
      0 -> domesticated translation (original_zh_text, official)
      1 -> foreignized translation (translated_text, re-translation / literal)
    """
    print(f"\n[1/4] Loading data from: {json_file} ...")
    if not os.path.exists(json_file):
        raise FileNotFoundError(
            f"{json_file} not found. Please run clean_json.py to generate cleaned.json first."
        )

    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"  Number of original triples: {len(data)}")

    texts: List[str] = []
    labels: List[int] = []

    skipped = 0
    for item in data:
        zh_official = (item.get("original_zh_text") or "").strip()
        zh_retrans = (item.get("translated_text") or "").strip()

        # Official translation -> domesticated
        if zh_official:
            texts.append(zh_official)
            labels.append(0)
        else:
            skipped += 1

        # Re-translation / MT -> foreignized
        if zh_retrans:
            texts.append(zh_retrans)
            labels.append(1)
        else:
            skipped += 1

    print(f"  Valid Chinese translation entries: {len(texts)}")
    if skipped > 0:
        print(f"  Skipped invalid / empty texts: {skipped}")

    return texts, labels


def encode_texts(model, texts: List[str], batch_size: int = 16) -> np.ndarray:
    """Encode texts into embeddings using Qwen3-Embedding."""
    print("\n[2/4] Generating text embeddings ...")
    import time

    t0 = time.time()
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=batch_size,
    )
    t1 = time.time()
    print(f"  Embedding shape: {embeddings.shape} (num_samples × dim)")
    print(
        f"  Encoding time: {t1-t0:.2f} seconds ({len(texts)/(t1-t0):.1f} samples/sec)"
    )
    return embeddings


def train_classifier(X_train: np.ndarray, y_train: List[int]) -> LogisticRegression:
    """Train a logistic regression classifier on top of embeddings."""
    print("\n[3/4] Training classifier (Logistic Regression) ...")
    clf = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced",  # simple handling for class imbalance
    )
    clf.fit(X_train, y_train)
    print("  ✓ Training completed!")
    return clf


def evaluate_classifier(
    clf: LogisticRegression, X_test: np.ndarray, y_test: List[int]
) -> None:
    """Evaluate classification performance on the test set."""
    print("\n[4/4] Evaluation on test set:")
    y_pred = clf.predict(X_test)
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Domesticated", "Foreignized"],
            digits=4,
        )
    )


def interactive_classification(model, clf: LogisticRegression) -> None:
    """CLI interaction: input a Chinese sentence and classify its style."""
    print("\n" + "=" * 80)
    print("Translation Style Classifier: Domesticated vs Foreignized")
    print("=" * 80)
    print("Instructions:")
    print("  - Input a Chinese translated sentence, press Enter,")
    print(
        "    the model will predict whether it is more 'domesticated' or 'foreignized'."
    )
    print("  - Type quit / exit / q to exit.")
    print("=" * 80)

    label_names = {
        0: "Domesticated translation (official style)",
        1: "Foreignized translation (re-translation / literal style)",
    }

    while True:
        try:
            text = input("\nPlease enter a Chinese translated sentence: ").strip()
            if text.lower() in ["quit", "exit", "q", ""]:
                print("\n👋 Bye!")
                break

            # Encode and predict
            emb = model.encode([text], convert_to_numpy=True)
            prob = clf.predict_proba(emb)[0]
            pred = int(np.argmax(prob))

            print("\n" + "-" * 80)
            print(f"Input text: {text}")
            print(f"Prediction: {label_names[pred]}")
            print(f"  P(Domesticated): {prob[0]:.4f} | P(Foreignized): {prob[1]:.4f}")
            print("-" * 80)
        except KeyboardInterrupt:
            print("\n\n👋 Bye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"results_classifier_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Setup file to save all output
    output_file = os.path.join(results_dir, "classifier_output.txt")
    log_file = open(output_file, "w", encoding="utf-8")
    
    # Custom print function that writes to both console and file
    original_print = print
    def print_and_log(*args, **kwargs):
        # Remove 'file' from kwargs if present, we'll set it explicitly
        kwargs_console = {k: v for k, v in kwargs.items() if k != 'file'}
        kwargs_file = kwargs_console.copy()
        kwargs_file['file'] = log_file
        
        original_print(*args, **kwargs_console)
        original_print(*args, **kwargs_file)
        log_file.flush()
    
    # Replace print temporarily
    import builtins
    builtins.print = print_and_log
    
    try:
        print("=" * 80)
        print("Translation Style Classifier based on Qwen3-Embedding")
        print("Task: distinguish 'Domesticated' vs 'Foreignized' translations")
        print("=" * 80)
        print(f"\nResults will be saved to: {results_dir}/")

        # 1. Load data
        texts, labels = load_translation_pairs("cleaned.json")

        # Use 2000 samples for training
        texts = texts[:2000]
        labels = labels[:2000]

        # Split into train / test
        X_train_texts, X_test_texts, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=0.2,
            random_state=42,
            stratify=labels,
        )
        print(
            f"\nDataset split: train={len(X_train_texts)} samples, "
            f"test={len(X_test_texts)} samples"
        )

        # 2. Load embedding model and generate vectors
        print("\nLoading Qwen3-Embedding model...")
        config.print_model_info()

        # Force CPU (GPU code commented out due to performance issues)
        device = "cpu"
        print(f"Using device: {device}")

        # # GPU auto-detection (commented out)
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # if device == "cuda":
        #     print(f"\nGPU detected: {torch.cuda.get_device_name(0)}")
        #     print(
        #         f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        #     )
        # print(f"Using device: {device}")

        model = config.load_model(device=device)

        # # GPU verification and warm-up (commented out)
        # if device == "cuda":
        #     print("\nVerifying GPU usage...")
        #     # ... GPU code ...

        batch_size = 16
        print(f"\nUsing batch_size={batch_size} for encoding")

        X_train_emb = encode_texts(model, X_train_texts, batch_size=batch_size)
        X_test_emb = encode_texts(model, X_test_texts, batch_size=batch_size)

        # 3. Train classifier
        clf = train_classifier(X_train_emb, y_train)

        # 4. Evaluate classifier
        evaluate_classifier(clf, X_test_emb, y_test)

        # 5. Save trained classifier
        model_save_path = os.path.join(results_dir, "style_classifier.joblib")
        print(f"\n[5/5] Saving classifier to: {model_save_path} ...")
        joblib.dump(clf, model_save_path)
        print(f"  ✓ Classifier saved successfully!")
        
        # Save classification report
        from save_results_helper import save_classification_report
        y_pred = clf.predict(X_test_emb)
        report_file = os.path.join(results_dir, "classification_report.txt")
        save_classification_report(y_test, y_pred, output_file=report_file)

        print(f"\n✓ All results saved to directory: {results_dir}/")
        print(f"  - classifier_output.txt (full console output)")
        print(f"  - style_classifier.joblib (trained model)")
        print(f"  - classification_report.txt (evaluation metrics)")

        # 6. Enter interactive mode
        interactive_classification(model, clf)
        
    finally:
        # Restore original print
        builtins.print = original_print
        log_file.close()
        
    finally:
        # Restore original print
        builtins.print = original_print
        log_file.close()


if __name__ == "__main__":
    main()
