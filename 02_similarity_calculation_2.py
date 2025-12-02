#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Game Localization Translation Strategy Analysis: Official Translation vs Re-translation
Based on Hollow Knight: Silksong translation data

This script analyzes semantic differences between official Chinese translations
and re-translated versions, exploring domestication/foreignization strategies
in game localization.
"""

from sentence_transformers import SentenceTransformer, util
import numpy as np
import json
import random
import config  # Import configuration file


def analyze_translation_triple(model, triple, chinese_anchors=None):
    """
    Analyze a single translation triple.

    Args:
        model: SentenceTransformer model
        triple: Dictionary containing 'original_en_text', 'original_zh_text', 'translated_text'
        chinese_anchors: Optional, list of common Chinese expressions (for evaluating localization)

    Returns:
        Dictionary containing various metrics
    """
    # Encode the triple
    texts = [
        triple.get("original_en_text", ""),  # English source
        triple.get("original_zh_text", ""),  # Official Chinese translation
        triple.get("translated_text", ""),  # Re-translation version
    ]
    embeddings = model.encode(texts, convert_to_tensor=True)

    # Calculate similarity matrix
    similarities = util.cos_sim(embeddings, embeddings)

    # Core metrics
    official_to_source = similarities[1][
        0
    ].item()  # Official translation vs source similarity
    retrans_to_source = similarities[2][0].item()  # Re-translation vs source similarity
    official_to_retrans = similarities[1][
        2
    ].item()  # Official vs re-translation similarity

    # Strategy difference (positive value means re-translation is closer to source)
    strategy_diff = retrans_to_source - official_to_source

    result = {
        "official_similarity": official_to_source,
        "retranslation_similarity": retrans_to_source,
        "inter_translation_similarity": official_to_retrans,
        "strategy_difference": strategy_diff,
    }

    # If Chinese anchor expressions are provided, calculate localization degree
    if chinese_anchors and len(chinese_anchors) > 0:
        anchor_embeddings = model.encode(chinese_anchors, convert_to_tensor=True)

        # Average similarity between official translation and Chinese common expressions
        official_to_anchors = util.cos_sim(embeddings[1:2], anchor_embeddings)
        official_localization = official_to_anchors.mean().item()

        # Average similarity between re-translation and Chinese common expressions
        retrans_to_anchors = util.cos_sim(embeddings[2:3], anchor_embeddings)
        retrans_localization = retrans_to_anchors.mean().item()

        result["official_localization"] = official_localization
        result["retranslation_localization"] = retrans_localization
        result["localization_difference"] = official_localization - retrans_localization

    return result


def print_triple_analysis(triple, result, index=None):
    """Print analysis results for a translation triple."""
    if index is not None:
        print(f"\n{'=' * 80}")
        print(f"Translation Comparison #{index}")
        print(f"{'=' * 80}")
    else:
        print(f"\n{'=' * 80}")
        print("Translation Comparison Analysis")
        print(f"{'=' * 80}")

    # Display texts (truncated)
    en_text = triple.get("original_en_text", "N/A")
    zh_official = triple.get("original_zh_text", "N/A")
    zh_retrans = triple.get("translated_text", "N/A")

    max_len = 100
    print(
        f"\nEN (Source):   {en_text[:max_len]}{'...' if len(en_text) > max_len else ''}"
    )
    print(
        f"ZH (Official):   {zh_official[:max_len]}{'...' if len(zh_official) > max_len else ''}"
    )
    print(
        f"ZH (Re-trans):   {zh_retrans[:max_len]}{'...' if len(zh_retrans) > max_len else ''}"
    )

    # Display analysis results
    print(f"\n{'-' * 80}")
    print("Semantic Similarity Analysis:")
    print(f"{'-' * 80}")

    official_sim = result["official_similarity"]
    retrans_sim = result["retranslation_similarity"]

    # Visualize similarity
    official_bar_length = int(official_sim * 40)
    retrans_bar_length = int(retrans_sim * 40)
    official_bar = "█" * official_bar_length + "░" * (40 - official_bar_length)
    retrans_bar = "█" * retrans_bar_length + "░" * (40 - retrans_bar_length)

    print(f"  Official -> Source: [{official_bar}] {official_sim:.4f}")
    print(f"  Re-trans -> Source: [{retrans_bar}] {retrans_sim:.4f}")
    print(f"  Official <-> Re-trans: {result['inter_translation_similarity']:.4f}")

    # Strategy difference interpretation
    strategy_diff = result["strategy_difference"]
    print(f"\n  Strategy Difference: {strategy_diff:+.4f}", end="")
    if strategy_diff > 0.05:
        print(" (Re-translation is closer to source)")
    elif strategy_diff < -0.05:
        print(" (Official translation is closer to source)")
    else:
        print(" (Both strategies are similar)")

    # Localization degree (if available)
    if "official_localization" in result:
        print(f"\n{'-' * 80}")
        print("Localization Degree Analysis:")
        print(f"{'-' * 80}")

        official_local = result["official_localization"]
        retrans_local = result["retranslation_localization"]
        local_diff = result["localization_difference"]

        official_local_bar_length = int(official_local * 40)
        retrans_local_bar_length = int(retrans_local * 40)
        official_local_bar = "█" * official_local_bar_length + "░" * (
            40 - official_local_bar_length
        )
        retrans_local_bar = "█" * retrans_local_bar_length + "░" * (
            40 - retrans_local_bar_length
        )

        print(f"  Official Localization: [{official_local_bar}] {official_local:.4f}")
        print(f"  Re-trans Localization: [{retrans_local_bar}] {retrans_local:.4f}")
        print(f"  Localization Difference: {local_diff:+.4f}", end="")
        if local_diff > 0.05:
            print(" (Official translation is more localized)")
        elif local_diff < -0.05:
            print(" (Re-translation is more localized)")
        else:
            print(" (Similar localization degree)")


def print_batch_summary(triples, results):
    """Print statistical summary of batch analysis."""
    print(f"\n{'=' * 80}")
    print(f"Batch Analysis Summary (Total: {len(results)} translation pairs)")
    print(f"{'=' * 80}")

    # Calculate averages
    avg_official_sim = np.mean([r["official_similarity"] for r in results])
    avg_retrans_sim = np.mean([r["retranslation_similarity"] for r in results])
    avg_strategy_diff = np.mean([r["strategy_difference"] for r in results])
    avg_inter_sim = np.mean([r["inter_translation_similarity"] for r in results])

    print(f"\nAverage Semantic Similarity:")
    print(f"  Official -> Source: {avg_official_sim:.4f}")
    print(f"  Re-trans -> Source: {avg_retrans_sim:.4f}")
    print(f"  Official <-> Re-trans: {avg_inter_sim:.4f}")
    print(f"  Average Strategy Difference: {avg_strategy_diff:+.4f}")

    # If localization data is available
    if "official_localization" in results[0]:
        avg_official_local = np.mean([r["official_localization"] for r in results])
        avg_retrans_local = np.mean([r["retranslation_localization"] for r in results])
        avg_local_diff = np.mean([r["localization_difference"] for r in results])

        print(f"\nAverage Localization Degree:")
        print(f"  Official Translation: {avg_official_local:.4f}")
        print(f"  Re-translation: {avg_retrans_local:.4f}")
        print(f"  Average Difference: {avg_local_diff:+.4f}")


def create_chinese_anchors():
    """Create Chinese common expression anchors (for evaluating localization degree)."""
    return [
        # Common daily expressions
        "今天天气很好",
        "我很高兴见到你",
        "这个问题很重要",
        "时间过得真快",
        "一切顺利",
        # Common idioms
        "一帆风顺",
        "马到成功",
        "心想事成",
        "事半功倍",
        # Common sentence patterns
        "这是一个重要的决定",
        "我们需要考虑这个问题",
        "情况比较复杂",
        "结果令人满意",
        # Game-related common expressions
        "勇敢的战士",
        "神秘的力量",
        "古老的传说",
        "危险的旅程",
    ]


def load_translation_data(filename, sample_size=None):
    """
    Load translation data.

    Args:
        filename: JSON file name
        sample_size: If specified, randomly sample the specified number of records

    Returns:
        List of translation data
    """
    print(f"Loading data: {filename}...")
    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"Total loaded: {len(data)} translation records")

    # Filter valid data
    valid_data = []
    for item in data:
        if (
            item.get("original_en_text")
            and item.get("original_zh_text")
            and item.get("translated_text")
        ):
            # Ensure texts are not empty
            if (
                item["original_en_text"].strip()
                and item["original_zh_text"].strip()
                and item["translated_text"].strip()
            ):
                valid_data.append(item)

    print(f"Valid records: {len(valid_data)}")

    # If sample size is specified
    if sample_size and sample_size < len(valid_data):
        print(f"Randomly sampling {sample_size} records...")
        valid_data = random.sample(valid_data, sample_size)

    return valid_data


def categorize_by_length(triples, results):
    """Categorize statistics by text length."""
    short = []  # < 50 characters
    medium = []  # 50-150 characters
    long = []  # > 150 characters

    for triple, result in zip(triples, results):
        en_len = len(triple["original_en_text"])
        if en_len < 50:
            short.append(result)
        elif en_len < 150:
            medium.append(result)
        else:
            long.append(result)

    print(f"\n{'-' * 80}")
    print("Statistics by Text Length:")
    print(f"{'-' * 80}")

    if short:
        avg_diff = np.mean([r["strategy_difference"] for r in short])
        print(
            f"  Short texts (<50 chars, n={len(short):3}): Avg strategy diff = {avg_diff:+.4f}"
        )

    if medium:
        avg_diff = np.mean([r["strategy_difference"] for r in medium])
        print(
            f"  Medium texts (50-150, n={len(medium):3}): Avg strategy diff = {avg_diff:+.4f}"
        )

    if long:
        avg_diff = np.mean([r["strategy_difference"] for r in long])
        print(
            f"  Long texts (>150 chars, n={len(long):3}): Avg strategy diff = {avg_diff:+.4f}"
        )


def main():
    print("=" * 80)
    print("Game Localization Translation Strategy Analysis: Official vs Re-translation")
    print("Data Source: Hollow Knight: Silksong")
    print("=" * 80)

    # 1. Load model
    print("\n[1/5] Loading model...")
    config.print_model_info()
    print()

    model = config.load_model(
        device="cpu"
    )  # Use CPU, change to 'cuda' if GPU available

    # 2. Load data
    print("\n[2/5] Loading translation data...")
    # Analyze 100 samples first, remove sample_size parameter to analyze all
    triples = load_translation_data("cleaned.json", sample_size=100)
    chinese_anchors = create_chinese_anchors()

    print(f"  Loaded {len(triples)} translation pairs")
    print(f"  Loaded {len(chinese_anchors)} Chinese common expression anchors")

    # 3. Analyze translation pairs
    print("\n[3/5] Analyzing translation pairs...")
    print("\n" + "▼" * 80)

    results = []
    display_count = 10  # Only show first 10 detailed results

    for i, triple in enumerate(triples, 1):
        # Analyze a single triple
        result = analyze_translation_triple(model, triple, chinese_anchors)
        results.append(result)

        # Only show first few detailed results
        if i <= display_count:
            print_triple_analysis(triple, result, index=i)

        # Progress indicator
        if i % 20 == 0:
            print(f"\n  Progress: {i}/{len(triples)}")

    if len(triples) > display_count:
        print(f"\n  ... Omitting {len(triples) - display_count} detailed results ...")

    print("\n" + "▲" * 80)

    # 4. Batch statistical summary
    print("\n[4/5] Generating statistical summary...")
    print_batch_summary(triples, results)

    # Statistics by length category
    categorize_by_length(triples, results)

    # 5. Key findings
    print(f"\n[5/5] Key Findings and Conclusions")
    print(f"{'=' * 80}")

    avg_strategy_diff = np.mean([r["strategy_difference"] for r in results])
    avg_inter_sim = np.mean([r["inter_translation_similarity"] for r in results])

    print("\nGame Localization Translation Analysis Conclusions:")

    if avg_strategy_diff > 0.05:
        print(
            f"  Re-translation is overall closer to English source (diff: {avg_strategy_diff:+.4f})"
        )
        print("  -> Re-translation may use more foreignization strategy (more literal)")
    elif avg_strategy_diff < -0.05:
        print(
            f"  Official translation is overall closer to English source (diff: {avg_strategy_diff:+.4f})"
        )
        print(
            "  -> Official translation may focus more on semantic accuracy than localization"
        )
    else:
        print(
            f"  Both strategies have similar semantic similarity (diff: {avg_strategy_diff:+.4f})"
        )

    print(f"\n  Average similarity (Official <-> Re-translation): {avg_inter_sim:.4f}")
    if avg_inter_sim > 0.85:
        print("  -> Both translations are generally consistent")
    elif avg_inter_sim > 0.70:
        print("  -> Both translations have some differences but share main semantics")
    else:
        print(
            "  -> Both translations differ significantly, strategies may be clearly different"
        )

    if "official_localization" in results[0]:
        avg_local_diff = np.mean([r["localization_difference"] for r in results])
        if avg_local_diff > 0.05:
            print(
                f"\n  Official translation has higher localization degree (diff: {avg_local_diff:+.4f})"
            )
            print("  -> Official translation is closer to common Chinese expressions")
        elif avg_local_diff < -0.05:
            print(
                f"\n  Re-translation has higher localization degree (diff: {avg_local_diff:+.4f})"
            )
            print("  -> Re-translation is closer to common Chinese expressions")

    print("\nResearch Recommendations:")
    print(
        "  1. Game localization needs to balance semantic accuracy and localized expression"
    )
    print(
        "  2. Further analyze translation strategies for specific text types (dialogue, narration, poetry)"
    )
    print(
        "  3. Translation strategies for proper nouns (character names, place names) deserve separate study"
    )
    print("  4. Increasing sample size can improve statistical reliability")

    print("\n" + "=" * 80)
    print("Analysis completed!")
    print("=" * 80)

    # Suggestions for extending analysis
    print("\nExtended Analysis Suggestions:")
    print(
        "  1. Modify sample_size parameter to analyze more data (e.g., 500, 1000 records)"
    )
    print("  2. Use grep tool to filter specific types of texts for analysis")
    print("  3. Combine with 04_text_clustering_visualization.py for visualization")
    print()


if __name__ == "__main__":
    main()
