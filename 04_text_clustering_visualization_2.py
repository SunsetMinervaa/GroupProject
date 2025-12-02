#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Semantic space comparison visualization: English source vs.
domesticated translation vs. foreignized translation.

Research question: which translation strategy is semantically closer
to the English source text?

This script places the English source, domesticated translation
(official Chinese), and foreignized translation (re-translation)
in the same embedding space, visualizes them with 2D and 3D scatter
plots, and computes the semantic distance between translations and
the source.
"""

import json
import random
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Tuple
import config


class TripleTranslationVisualizer:
    """Visualization tool for comparing translation triples."""

    def __init__(self):
        """Initialize visualizer and model."""
        print("=" * 80)
        print("Semantic Space Comparison: English vs Domesticated vs Foreignized")
        print("=" * 80)
        print("\nLoading Qwen3-Embedding model...")
        self.model = config.load_model(device="cpu")

        self.texts = []
        self.labels = []
        self.embeddings = None
        self.reduced_embeddings = None

    def load_data(self, json_file: str, sample_size: int = 200):
        """
        Load and prepare triple data.

        Args:
            json_file: path to cleaned.json
            sample_size: number of triples to sample
                (each triple contains 3 texts: English + domesticated + foreignized)
        """
        print(f"\nLoading data from: {json_file}...")
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"Total records: {len(data)}")

        # Random sampling
        if sample_size and sample_size < len(data):
            print(f"Randomly sampling {sample_size} triples...")
            data = random.sample(data, sample_size)

        # Extract three kinds of texts
        texts = []
        labels = []

        for item in data:
            # English source
            texts.append(item["original_en_text"])
            labels.append("English Source")

            # Domesticated translation (official Chinese)
            texts.append(item["original_zh_text"])
            labels.append("Domesticated Translation")

            # Foreignized translation (re-translation)
            texts.append(item["translated_text"])
            labels.append("Foreignized Translation")

        self.texts = texts
        self.labels = labels

        print(f"\nPrepared {len(texts)} texts ({len(texts)//3} translation triples)")
        print(f"  - English Source: {labels.count('English Source')}")
        print(
            f"  - Domesticated Translation: {labels.count('Domesticated Translation')}"
        )
        print(f"  - Foreignized Translation: {labels.count('Foreignized Translation')}")

    def generate_embeddings(self):
        """Generate embeddings for all texts."""
        print("\nGenerating embeddings...")
        self.embeddings = self.model.encode(
            self.texts, show_progress_bar=True, convert_to_numpy=True
        )
        print(f"✓ Embeddings generated! Shape: {self.embeddings.shape}")

    def reduce_dimensions_2d(self, perplexity: int = 30):
        """Use t-SNE to reduce embeddings to 2D."""
        print(f"\nRunning t-SNE to reduce to 2D...")
        print(f"Parameters: perplexity={perplexity}, random_state=42")

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
            max_iter=1000,
            verbose=1,
        )

        self.reduced_embeddings = tsne.fit_transform(self.embeddings)
        print(f"✓ t-SNE 2D reduction completed! Shape: {self.reduced_embeddings.shape}")

        return self.reduced_embeddings

    def reduce_dimensions_3d(self, perplexity: int = 30):
        """Use t-SNE to reduce embeddings to 3D."""
        print(f"\nRunning t-SNE to reduce to 3D...")
        print(f"Parameters: perplexity={perplexity}, random_state=42")

        tsne = TSNE(
            n_components=3,
            perplexity=perplexity,
            random_state=42,
            max_iter=1000,
            verbose=1,
        )

        self.reduced_embeddings = tsne.fit_transform(self.embeddings)
        print(f"✓ t-SNE 3D reduction completed! Shape: {self.reduced_embeddings.shape}")

        return self.reduced_embeddings

    def visualize_2d(self, save_path: str = "translation_comparison_2d.html"):
        """Create a 2D visualization."""
        if self.reduced_embeddings is None or self.reduced_embeddings.shape[1] != 2:
            raise ValueError(
                "Please call reduce_dimensions_2d() before visualize_2d()."
            )

        print(f"\nCreating 2D visualization...")

        fig = go.Figure()

        # First, add triangle boundaries between triples
        print("  Adding triangle boundaries for each triple...")
        for i in range(0, len(self.texts), 3):
            # Get coordinates of the three points
            en_x, en_y = self.reduced_embeddings[i]  # English
            dom_x, dom_y = self.reduced_embeddings[i + 1]  # Domesticated
            for_x, for_y = self.reduced_embeddings[i + 2]  # Foreignized

            # Draw triangle boundary (closed path)
            fig.add_trace(
                go.Scatter(
                    x=[en_x, dom_x, for_x, en_x],  # 闭合三角形
                    y=[en_y, dom_y, for_y, en_y],
                    mode="lines",
                    line=dict(color="rgba(150, 150, 150, 0.3)", width=1, dash="dot"),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Define colors and markers
        style_config = {
            "English Source": {
                "color": "#2ecc71",
                "symbol": "circle",
                "size": 10,
            },  # green
            "Domesticated Translation": {
                "color": "#3498db",
                "symbol": "square",
                "size": 9,
            },  # blue
            "Foreignized Translation": {
                "color": "#e74c3c",
                "symbol": "diamond",
                "size": 9,
            },  # red
        }

        # Create one trace for each label type
        for label_type in [
            "English Source",
            "Domesticated Translation",
            "Foreignized Translation",
        ]:
            indices = [i for i, l in enumerate(self.labels) if l == label_type]

            # Prepare hover texts
            hover_texts = [
                f"<b>{label_type}</b><br>"
                + f"{self.texts[i][:100]}{'...' if len(self.texts[i]) > 100 else ''}"
                for i in indices
            ]

            config_style = style_config[label_type]

            fig.add_trace(
                go.Scatter(
                    x=self.reduced_embeddings[indices, 0],
                    y=self.reduced_embeddings[indices, 1],
                    mode="markers",
                    name=label_type,
                    marker=dict(
                        size=config_style["size"],
                        color=config_style["color"],
                        opacity=0.8,
                        symbol=config_style["symbol"],
                        line=dict(width=1.5, color="white"),
                    ),
                    text=hover_texts,
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        fig.update_layout(
            title=dict(
                text="Semantic Space: English vs Domesticated vs Foreignized (2D)<br>"
                + "<sub>Observe which translation strategy is closer to the English source</sub>",
                font=dict(size=20, family="Arial, sans-serif"),
            ),
            xaxis_title="Semantic Dimension 1",
            yaxis_title="Semantic Dimension 2",
            hovermode="closest",
            width=1400,
            height=900,
            template="plotly_white",
            legend=dict(title="Text Type", font=dict(size=14), x=0.02, y=0.98),
        )

        # Save
        print(f"  Saving to: {save_path}")
        html_str = fig.to_html(
            include_plotlyjs="cdn", config={"displayModeBar": True, "responsive": True}
        )
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_str)

        print(f"✓ 2D visualization saved.")

        return fig

    def visualize_3d(self, save_path: str = "translation_comparison_3d.html"):
        """Create a 3D visualization."""
        if self.reduced_embeddings is None or self.reduced_embeddings.shape[1] != 3:
            raise ValueError(
                "Please call reduce_dimensions_3d() before visualize_3d()."
            )

        print(f"\nCreating 3D visualization...")

        fig = go.Figure()

        # First, add triangle edges between triples
        print("  Adding triangle edges for each triple...")
        for i in range(0, len(self.texts), 3):
            # Get coordinates of the three points
            en_x, en_y, en_z = self.reduced_embeddings[i]  # English
            dom_x, dom_y, dom_z = self.reduced_embeddings[i + 1]  # Domesticated
            for_x, for_y, for_z = self.reduced_embeddings[i + 2]  # Foreignized

            # Draw the three edges of the triangle
            # Edge 1: English -> Domesticated
            fig.add_trace(
                go.Scatter3d(
                    x=[en_x, dom_x],
                    y=[en_y, dom_y],
                    z=[en_z, dom_z],
                    mode="lines",
                    line=dict(color="rgba(150, 150, 150, 0.3)", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Edge 2: Domesticated -> Foreignized
            fig.add_trace(
                go.Scatter3d(
                    x=[dom_x, for_x],
                    y=[dom_y, for_y],
                    z=[dom_z, for_z],
                    mode="lines",
                    line=dict(color="rgba(150, 150, 150, 0.3)", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            # Edge 3: Foreignized -> English
            fig.add_trace(
                go.Scatter3d(
                    x=[for_x, en_x],
                    y=[for_y, en_y],
                    z=[for_z, en_z],
                    mode="lines",
                    line=dict(color="rgba(150, 150, 150, 0.3)", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Define colors and markers
        style_config = {
            "English Source": {
                "color": "#2ecc71",
                "symbol": "circle",
                "size": 6,
            },  # green
            "Domesticated Translation": {
                "color": "#3498db",
                "symbol": "square",
                "size": 5,
            },  # blue
            "Foreignized Translation": {
                "color": "#e74c3c",
                "symbol": "diamond",
                "size": 5,
            },  # red
        }

        # Create one trace for each label type
        for label_type in [
            "English Source",
            "Domesticated Translation",
            "Foreignized Translation",
        ]:
            indices = [i for i, l in enumerate(self.labels) if l == label_type]

            # Prepare hover texts
            hover_texts = [
                f"<b>{label_type}</b><br>"
                + f"{self.texts[i][:100]}{'...' if len(self.texts[i]) > 100 else ''}"
                for i in indices
            ]

            config_style = style_config[label_type]

            fig.add_trace(
                go.Scatter3d(
                    x=self.reduced_embeddings[indices, 0],
                    y=self.reduced_embeddings[indices, 1],
                    z=self.reduced_embeddings[indices, 2],
                    mode="markers",
                    name=label_type,
                    marker=dict(
                        size=config_style["size"],
                        color=config_style["color"],
                        opacity=0.8,
                        symbol=config_style["symbol"],
                        line=dict(width=1, color="white"),
                    ),
                    text=hover_texts,
                    hovertemplate="%{text}<extra></extra>",
                )
            )

        fig.update_layout(
            title=dict(
                text="Semantic Space: English vs Domesticated vs Foreignized (3D)<br>"
                + "<sub>Rotate and zoom to explore spatial relationships</sub>",
                font=dict(size=20, family="Arial, sans-serif"),
            ),
            scene=dict(
                xaxis_title="Semantic Dimension 1",
                yaxis_title="Semantic Dimension 2",
                zaxis_title="Semantic Dimension 3",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            hovermode="closest",
            width=1400,
            height=900,
            template="plotly_white",
            legend=dict(title="Text Type", font=dict(size=14)),
        )

        # Save
        print(f"  Saving to: {save_path}")
        html_str = fig.to_html(
            include_plotlyjs="cdn", config={"displayModeBar": True, "responsive": True}
        )
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(html_str)

        print(f"✓ 3D visualization saved.")

        return fig

    def calculate_distances(self):
        """Compute average distances from translations to the English source."""
        if self.reduced_embeddings is None:
            raise ValueError("Please run a dimensionality reduction method first.")

        print("\n" + "=" * 80)
        print("Distance Analysis: Semantic distances to English source")
        print("=" * 80)

        distances_dom = []  # Domesticated -> English
        distances_for = []  # Foreignized -> English

        # Every 3 texts form a triple (English, Domesticated, Foreignized)
        for i in range(0, len(self.texts), 3):
            en_vec = self.reduced_embeddings[i]  # English
            dom_vec = self.reduced_embeddings[i + 1]  # Domesticated
            for_vec = self.reduced_embeddings[i + 2]  # Foreignized

            # Euclidean distance
            dist_dom = np.linalg.norm(en_vec - dom_vec)
            dist_for = np.linalg.norm(en_vec - for_vec)

            distances_dom.append(dist_dom)
            distances_for.append(dist_for)

        avg_dist_dom = np.mean(distances_dom)
        avg_dist_for = np.mean(distances_for)

        print(f"\nAverage distance (Domesticated -> English): {avg_dist_dom:.4f}")
        print(f"Average distance (Foreignized -> English): {avg_dist_for:.4f}")
        print(
            f"Difference (Foreignized - Domesticated): {avg_dist_for - avg_dist_dom:+.4f}"
        )

        if avg_dist_dom < avg_dist_for:
            print(
                "\nConclusion: Domesticated translations are closer to the English source."
            )
            print(
                "  -> The domestication strategy may better preserve the core semantics of the source text."
            )
        elif avg_dist_for < avg_dist_dom:
            print(
                "\nConclusion: Foreignized translations are closer to the English source."
            )
            print(
                "  -> The foreignization strategy may be more faithful to the source expression."
            )
        else:
            print(
                "\nConclusion: Both strategies have similar distances to the English source."
            )

        # Descriptive statistics
        print("\nDetailed statistics:")
        print(
            f"  Domesticated distances - min: {np.min(distances_dom):.4f}, "
            f"max: {np.max(distances_dom):.4f}, std: {np.std(distances_dom):.4f}"
        )
        print(
            f"  Foreignized distances - min: {np.min(distances_for):.4f}, "
            f"max: {np.max(distances_for):.4f}, std: {np.std(distances_for):.4f}"
        )

        return {
            "domesticated_avg": avg_dist_dom,
            "foreignized_avg": avg_dist_for,
            "difference": avg_dist_for - avg_dist_dom,
            "domesticated_distances": distances_dom,
            "foreignized_distances": distances_for,
        }


def main():
    """Main entry point."""
    # 1. Initialize visualizer
    visualizer = TripleTranslationVisualizer()

    # 2. Load data (sample 30 triples, 90 texts)
    visualizer.load_data("cleaned.json", sample_size=30)

    # 3. Generate embeddings
    visualizer.generate_embeddings()

    # 4. 2D visualization
    print("\n" + "▼" * 80)
    print("Part 1: 2D Visualization")
    print("▼" * 80)

    visualizer.reduce_dimensions_2d(perplexity=30)
    visualizer.visualize_2d("translation_comparison_2d.html")

    # Compute distances in 2D space
    stats_2d = visualizer.calculate_distances()

    # 5. 3D visualization
    print("\n" + "▼" * 80)
    print("Part 2: 3D Visualization")
    print("▼" * 80)

    visualizer.reduce_dimensions_3d(perplexity=30)
    visualizer.visualize_3d("translation_comparison_3d.html")

    # Compute distances in 3D space
    stats_3d = visualizer.calculate_distances()

    # 6. Summary
    print("\n" + "=" * 80)
    print("✓ Analysis completed!")
    print("=" * 80)

    print("\nGenerated files:")
    print("  1. translation_comparison_2d.html - 2D scatter plot")
    print("  2. translation_comparison_3d.html - 3D scatter plot (rotatable)")

    print("\nHow to use:")
    print("  - Open the HTML files in a browser.")
    print("  - Hover over points to see the underlying text.")
    print("  - In the 3D plot, drag to rotate and scroll to zoom.")
    print("  - Click legend entries to hide/show specific text types.")

    print("\nWhat to observe:")
    print("  - Green circles = English source")
    print("  - Blue squares  = Domesticated translations (official Chinese)")
    print("  - Red diamonds  = Foreignized translations (re-translations)")
    print("  - Gray dashed triangles = boundaries of each triple")
    print("  - Which color (blue/red) tends to be closer to green.")
    print("  - Flatter triangles indicate smaller differences between strategies.")

    print("\nResearch suggestions:")
    print("  1. Check whether the three text types form distinct clusters.")
    print("  2. Identify translation triples that are particularly close or far apart.")
    print("  3. Analyze spatial patterns of domestication vs foreignization.")
    print(
        "  4. Combine visualization with textual analysis to interpret distance differences."
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
