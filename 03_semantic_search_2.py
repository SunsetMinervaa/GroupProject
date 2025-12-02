#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized semantic search for translation triples.
Improvement: precompute all text embeddings to greatly speed up search.
"""

import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import config
from typing import List, Dict
import torch


class TranslationTripleSearcher:
    """Optimized semantic search tool for translation triples."""

    def __init__(self, json_file: str = "cleaned.json"):
        """Initialize and precompute embeddings."""
        print("=" * 80)
        print("Optimized Semantic Search for Translation Triples")
        print("=" * 80)
        print("\nLoading Qwen3-Embedding model...")
        # Automatically detect and use available GPU (if any)
        import torch

        # Diagnose CUDA environment
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU device: {torch.cuda.get_device_name(0)}")
            print(
                f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
            )
        else:
            print("⚠️  No GPU detected, falling back to CPU (may be slower).")
            print(
                "   If you have an NVIDIA GPU, please install the CUDA version of PyTorch."
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\nUsing device: {device}")
        self.model = config.load_model(device=device)

        # Load translation data (only first N triples)
        print(f"\nLoading translation data from: {json_file}...")
        with open(json_file, "r", encoding="utf-8") as f:
            all_triples = json.load(f)

        # Use only the first N triples for precomputation (speed up)
        max_triples = 100
        self.triples = all_triples[:max_triples]
        print(
            f"✓ Data loaded! Using the first {len(self.triples)} translation triples for analysis."
        )
        print(
            f"   (There are {len(all_triples)} triples in total; only the first {max_triples} are used for speed.)"
        )

        # Prepare all texts (for fast search)
        self.all_texts = []
        self.text_to_triple = {}  # mapping from text to its triple

        for triple in self.triples:
            # English source
            self.all_texts.append(triple["original_en_text"])
            self.text_to_triple[triple["original_en_text"]] = triple

            # Domesticated translation (official Chinese)
            self.all_texts.append(triple["original_zh_text"])
            self.text_to_triple[triple["original_zh_text"]] = triple

            # Foreignized translation (re-translation)
            self.all_texts.append(triple["translated_text"])
            self.text_to_triple[triple["translated_text"]] = triple

        print(f"Prepared {len(self.all_texts)} texts for search...")

        # Precompute embeddings (fast initialization)
        print(f"\nPrecomputing embeddings for {len(self.all_texts)} texts...")
        print("Note: This takes about 30–60 seconds on CPU or 5–10 seconds on GPU.")
        self.text_embeddings = self.model.encode(
            self.all_texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,
        )
        print("✓ Text embeddings precomputation completed!")

    def search_by_keyword(
        self, keyword: str, similarity_threshold: float = 0.5, top_n: int = 10
    ):
        """
        Search for related translation triples based on a keyword
        (uses precomputed embeddings for fast search).
        """
        print(f'\nSearch keyword: "{keyword}"')
        print(f"Similarity threshold: {similarity_threshold}")

        # 生成关键词向量
        keyword_embedding = self.model.encode(keyword, convert_to_tensor=True)

        # Compute similarity (using precomputed embeddings)
        similarities = util.cos_sim(keyword_embedding, self.text_embeddings)[0]

        # Find texts above threshold
        above_threshold_indices = torch.nonzero(
            similarities > similarity_threshold
        ).squeeze()
        above_threshold_count = len(above_threshold_indices)

        print(
            f"\nFound {above_threshold_count} texts with similarity > {similarity_threshold}"
        )

        # Get all related triples (deduplicated)
        related_triples = set()
        for idx in above_threshold_indices:
            text = self.all_texts[idx]
            triple = self.text_to_triple[text]
            # Use English source text as the unique identifier
            related_triples.add(triple["original_en_text"])

        print(f"Involving {len(related_triples)} translation triples")

        # Compute max similarity for each triple
        triple_scores = []
        for en_text in related_triples:
            triple = self.text_to_triple[en_text]

            # Get indices in precomputed embeddings
            en_idx = self.all_texts.index(triple["original_en_text"])
            dom_idx = self.all_texts.index(triple["original_zh_text"])
            for_idx = self.all_texts.index(triple["translated_text"])

            # Get similarities from precomputed embeddings
            en_sim = similarities[en_idx].item()
            dom_sim = similarities[dom_idx].item()
            for_sim = similarities[for_idx].item()

            # Use the maximum similarity of the three as the triple's score
            max_sim = max(en_sim, dom_sim, for_sim)
            triple_scores.append((triple, max_sim))

        # 按相似度排序
        triple_scores.sort(key=lambda x: x[1], reverse=True)

        above_threshold_triples = triple_scores

        # Output top-n most relevant triples
        print(
            f"\nTop {min(top_n, len(above_threshold_triples))} related translation triples:"
        )
        print("=" * 100)

        for i, (triple, score) in enumerate(above_threshold_triples[:top_n], 1):
            print(f"\n{i}. Similarity: {score:.4f}")
            print(
                f"  English source: {triple['original_en_text'][:100]}{'...' if len(triple['original_en_text']) > 100 else ''}"
            )
            print(
                f"  Domesticated translation: {triple['original_zh_text'][:100]}{'...' if len(triple['original_zh_text']) > 100 else ''}"
            )
            print(
                f"  Foreignized translation: {triple['translated_text'][:100]}{'...' if len(triple['translated_text']) > 100 else ''}"
            )
            print("-" * 100)


def main():
    """Main entry point."""
    # 1. Initialize searcher (precompute embeddings)
    print("Initializing searcher...")
    print(
        "Note: Precomputing embeddings for the first 100 translation triples may take 30–60 seconds."
    )
    searcher = TranslationTripleSearcher()

    # 2. Interactive search (using precomputed embeddings, very fast)
    print("\n" + "=" * 80)
    print("Interactive Search (type 'quit' or 'exit' to exit)")
    print("Note: Search is very fast (<1 second), based on precomputed embeddings.")
    print("=" * 80)

    while True:
        try:
            keyword = input("\nPlease enter a search keyword: ").strip()

            if keyword.lower() in ["quit", "exit", "q"]:
                print("Thank you for using the tool! Bye!")
                break

            if not keyword:
                print("Keyword cannot be empty, please try again.")
                continue

            # Get similarity threshold (optional)
            threshold_str = input(
                "Please enter similarity threshold (default 0.5): "
            ).strip()
            threshold = 0.5 if not threshold_str else float(threshold_str)

            # Perform search (using precomputed embeddings)
            searcher.search_by_keyword(keyword, similarity_threshold=threshold)

        except KeyboardInterrupt:
            print("\n\nProgram interrupted, thank you for using the tool!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
