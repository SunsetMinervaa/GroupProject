#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for saving analysis results to files.

Usage:
    from save_results_helper import save_analysis_results, save_classification_report
    
    # Save similarity analysis results
    save_analysis_results(results, triples, output_file="results.json")
    
    # Save classification report
    save_classification_report(y_test, y_pred, output_file="classification_report.txt")
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any
import numpy as np


def save_analysis_results(
    results: List[Dict],
    triples: List[Dict],
    output_file: str = "analysis_results.json",
    include_texts: bool = True,
):
    """
    Save similarity analysis results to JSON file.
    
    Args:
        results: List of analysis result dictionaries
        triples: List of translation triples
        output_file: Output file path
        include_texts: Whether to include full text content
    """
    output_data = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(results),
            "analysis_type": "translation_strategy_similarity",
        },
        "summary": {
            "avg_official_similarity": float(np.mean([r["official_similarity"] for r in results])),
            "avg_retranslation_similarity": float(np.mean([r["retranslation_similarity"] for r in results])),
            "avg_strategy_difference": float(np.mean([r["strategy_difference"] for r in results])),
            "avg_inter_translation_similarity": float(np.mean([r["inter_translation_similarity"] for r in results])),
        },
        "results": [],
    }
    
    # Add localization summary if available
    if "official_localization" in results[0]:
        output_data["summary"]["avg_official_localization"] = float(
            np.mean([r["official_localization"] for r in results])
        )
        output_data["summary"]["avg_retranslation_localization"] = float(
            np.mean([r["retranslation_localization"] for r in results])
        )
        output_data["summary"]["avg_localization_difference"] = float(
            np.mean([r["localization_difference"] for r in results])
        )
    
    # Add individual results
    for triple, result in zip(triples, results):
        item = {
            "metrics": {
                "official_similarity": float(result["official_similarity"]),
                "retranslation_similarity": float(result["retranslation_similarity"]),
                "inter_translation_similarity": float(result["inter_translation_similarity"]),
                "strategy_difference": float(result["strategy_difference"]),
            },
        }
        
        if include_texts:
            item["texts"] = {
                "english_source": triple.get("original_en_text", ""),
                "official_translation": triple.get("original_zh_text", ""),
                "re_translation": triple.get("translated_text", ""),
            }
        
        if "official_localization" in result:
            item["metrics"]["official_localization"] = float(result["official_localization"])
            item["metrics"]["retranslation_localization"] = float(result["retranslation_localization"])
            item["metrics"]["localization_difference"] = float(result["localization_difference"])
        
        output_data["results"].append(item)
    
    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Analysis results saved to: {output_file}")
    print(f"  Total records: {len(results)}")
    print(f"  File size: {os.path.getsize(output_file) / 1024:.2f} KB")


def save_classification_report(
    y_test: List[int],
    y_pred: List[int],
    output_file: str = "classification_report.txt",
    target_names: List[str] = None,
):
    """
    Save classification report to text file.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        output_file: Output file path
        target_names: Class names (default: ["Domesticated", "Foreignized"])
    """
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    
    if target_names is None:
        target_names = ["Domesticated", "Foreignized"]
    
    report = classification_report(y_test, y_pred, target_names=target_names, digits=4)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Classification Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Total samples: {len(y_test)}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{'':20} Predicted\n")
        f.write(f"{'':20} {target_names[0]:15} {target_names[1]:15}\n")
        f.write(f"Actual {target_names[0]:15} {cm[0][0]:15} {cm[0][1]:15}\n")
        f.write(f"      {target_names[1]:15} {cm[1][0]:15} {cm[1][1]:15}\n\n")
        f.write(report)
    
    print(f"\n✓ Classification report saved to: {output_file}")


def save_embeddings(
    embeddings: np.ndarray,
    texts: List[str],
    labels: List[str] = None,
    output_file: str = "embeddings.npz",
):
    """
    Save embeddings to compressed numpy file.
    
    Args:
        embeddings: Embedding vectors (n_samples, n_dim)
        texts: Corresponding texts
        labels: Optional labels for each text
        output_file: Output file path (.npz format)
    """
    save_dict = {
        "embeddings": embeddings,
        "texts": texts,
    }
    
    if labels is not None:
        save_dict["labels"] = labels
    
    np.savez_compressed(output_file, **save_dict)
    print(f"\n✓ Embeddings saved to: {output_file}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")


def load_embeddings(input_file: str = "embeddings.npz"):
    """
    Load embeddings from compressed numpy file.
    
    Args:
        input_file: Input file path
        
    Returns:
        Dictionary containing embeddings, texts, and optionally labels
    """
    data = np.load(input_file, allow_pickle=True)
    result = {
        "embeddings": data["embeddings"],
        "texts": data["texts"].tolist(),
    }
    
    if "labels" in data:
        result["labels"] = data["labels"].tolist()
    
    print(f"✓ Loaded embeddings from: {input_file}")
    print(f"  Shape: {result['embeddings'].shape}")
    return result


def create_results_directory(base_name: str = "results") -> str:
    """
    Create a timestamped results directory.
    
    Args:
        base_name: Base name for the directory
        
    Returns:
        Path to created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{base_name}_{timestamp}"
    os.makedirs(dir_name, exist_ok=True)
    print(f"✓ Created results directory: {dir_name}")
    return dir_name

