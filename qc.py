import marimo

__generated_with = "0.14.12"
app = marimo.App(width="full")


@app.cell
def _():
    import scanpy as sc
    import matplotlib.pyplot as plt
    import seaborn as sns
    import marimo as mo
    import pandas as pd
    import numpy as np
    return mo, plt, sc, sns


@app.cell
def _(mo):
    mo.md(
        r"""
    # Single-Cell QC Analysis Tool

    Interactive tool for quality control analysis of single-cell RNA-seq data.
    Select QC metrics, adjust thresholds, and visualize results on UMAP
    """
    )
    return


@app.cell
def _(plt, sc, sns):
    # QC Analysis Functions
    qc_vars_default = [
        "Doublet score",
        "Fraction mitochondrial UMIs", 
        "Genes detected"
    ]

    def create_qc_dataframe(adata):
        """Convert AnnData to DataFrame for interactive plotting (cell-level)"""
        import pandas as pd

        # Get QC metrics and metadata
        qc_columns = ['leiden', 'supertype', 'supertype_scANVI'] + qc_vars_default

        # Filter to only include columns that exist in the adata object
        available_columns = [col for col in qc_columns if col in adata.obs.columns]

        plot_df = adata.obs[available_columns].copy()

        # Add UMAP coordinates if available
        if 'X_umap' in adata.obsm:
            umap_coords = pd.DataFrame(
                adata.obsm['X_umap'], 
                columns=['UMAP1', 'UMAP2'],
                index=adata.obs.index
            )
            plot_df = pd.concat([plot_df, umap_coords], axis=1)

        return plot_df


    def plot_QC_metric_hist(adata, qc_metric):
        """Return a matplotlib figure showing the distribution of per-cluster means for `qc_metric`."""
        if qc_metric not in adata.obs.columns:
            return None

        # Compute mean value per Leiden cluster
        cluster_means = (
            adata.obs.loc[:, ["leiden", qc_metric]]
            .groupby("leiden", observed=True)
            .mean()
            .loc[:, qc_metric]
        )

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.histplot(cluster_means.values, ax=ax, color="steelblue", edgecolor="black")
        ax.set_xlabel(qc_metric)
        ax.set_ylabel("Number of Clusters")
        ax.set_title(f"Distribution of {qc_metric} across Leiden clusters")
        fig.tight_layout()

        return fig

    def flag_and_plot_QC(adata, qc_thresh, qc_vars, qc_dir):
        """Flag cells based on QC thresholds and plot results"""
        adata_copy = adata.copy()

        for l, k in enumerate(qc_vars):
            tmp = (
                adata_copy.obs.loc[:, ["leiden", k]]
                .groupby("leiden", observed=True)
                .mean()
                .loc[:, k]
                .to_dict()
            )
            groups = []
            for i, j in tmp.items():
                if qc_dir[l] == "gt":
                    if j > qc_thresh[l]:
                        groups.append(i)
                else:
                    if j < qc_thresh[l]:
                        groups.append(i)

            # Flag clusters that fail QC
            adata_copy.obs[f"cluster_{k}_flag"] = "False"
            adata_copy.obs.loc[
                adata_copy.obs["leiden"].isin(groups), f"cluster_{k}_flag"
            ] = "True"

        return adata_copy, groups

    def create_umap_plots(adata, qc_metric, threshold, direction):
        """Create UMAP plots showing QC flagging results"""
        # Create temporary qc_vars, qc_thresh, qc_dir for the selected metric
        if qc_metric == "Doublet score":
            qc_vars = ["Doublet score"]
            qc_thresh = [threshold]
            qc_dir = ["gt"]  # greater than threshold is bad
        elif qc_metric == "Genes detected":
            qc_vars = ["Genes detected"] 
            qc_thresh = [threshold]
            qc_dir = ["lt"]  # less than threshold is bad
        else:  # Fraction mitochondrial UMIs
            qc_vars = ["Fraction mitochondrial UMIs"]
            qc_thresh = [threshold]
            qc_dir = ["gt"]  # greater than threshold is bad

        adata_flagged, flagged_groups = flag_and_plot_QC(adata, qc_thresh, qc_vars, qc_dir)

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Plot 1: Leiden clusters with flagged ones highlighted
        if flagged_groups:
            sc.pl.umap(
                adata_flagged,
                color=["leiden"],
                legend_loc="on data",
                frameon=False,
                groups=flagged_groups,
                na_in_legend=False,
                sort_order=False,
                ax=axes[0,0],
                show=False
            )
            axes[0,0].set_title(f"Flagged Clusters: {qc_metric} (highlighted)")
        else:
            sc.pl.umap(
                adata_flagged,
                color=["leiden"],
                legend_loc="on data",
                frameon=False,
                ax=axes[0,0],
                show=False
            )
            axes[0,0].set_title(f"All Clusters: {qc_metric} (none flagged)")

        # Plot 2: QC flag coloring
        flag_col = f"cluster_{qc_metric}_flag"
        if flag_col in adata_flagged.obs.columns:
            sc.pl.umap(
                adata_flagged,
                color=flag_col,
                frameon=False,
                palette={"True": "red", "False": "lightgrey"},
                sort_order=False,
                ax=axes[0,1],
                show=False
            )
        axes[0,1].set_title("QC Flagged Cells (Red = Flagged)")

        # Plot 3: Continuous QC metric values
        sc.pl.umap(
            adata_flagged,
            color=qc_metric,
            frameon=False,
            cmap="viridis",
            sort_order=False,
            ax=axes[1,0],
            show=False
        )
        axes[1,0].set_title(f"{qc_metric} Values")

        # Plot 4: Supertype for reference
        sc.pl.umap(
            adata_flagged,
            color="supertype",
            frameon=False,
            ax=axes[1,1],
            show=False
        )
        axes[1,1].set_title("Cell Supertypes")

        plt.tight_layout()

        # Return the figure object along with the processed data
        return fig, adata_flagged, flagged_groups

    return create_umap_plots, plot_QC_metric_hist, qc_vars_default


@app.cell
def _(mo):
    # Data loading controls
    data_path = mo.ui.text(
        label="Path to .h5ad file",
        value="test_data/All_scANVI.2025-05-13.h5ad",
        full_width=True
    )

    resolution_slider = mo.ui.slider(
        start=0.1, 
        stop=10.0, 
        value=5.0, 
        step=0.1, 
        label="Leiden Resolution"
    )

    mo.vstack([
        mo.md("**Data Configuration:**"),
        data_path,
        resolution_slider
    ])
    return data_path, resolution_slider


@app.cell
def _(data_path, mo, sc):
    # Load and preprocess data
    mo.md(f"Loading data from: `{data_path.value}`")
    adata = sc.read_h5ad(data_path.value)

    mo.md(f"""
    **Data loaded successfully!**
    - Shape: {adata.shape[0]} cells × {adata.shape[1]} genes
    """)
    return (adata,)


@app.cell
def _(adata, mo, resolution_slider, sc):
    # Preprocessing pipeline
    mo.md("**Preprocessing data...**")

    # Ensure the required representation and neighbors are present
    if "X_scVI" not in adata.obsm:
        mo.md(" Warning: X_scVI representation not found")

    # Compute neighbors if not present
    if "neighbors" not in adata.uns:
        mo.md("Computing neighbors using X_scVI representation...")
        sc.pp.neighbors(adata, use_rep="X_scVI")

    # Compute UMAP if not present
    if "X_umap" not in adata.obsm:
        mo.md("Computing UMAP...")
        sc.tl.umap(adata, min_dist=0.3)

    # Perform Leiden clustering
    mo.md(f"Performing Leiden clustering with resolution={resolution_slider.value}")
    sc.tl.leiden(adata, resolution=resolution_slider.value, random_state=1, flavor="igraph", n_iterations=2, directed=False)

    n_clusters = len(adata.obs['leiden'].unique())
    mo.md(f"Preprocessing complete: Found {n_clusters} Leiden clusters")
    return


@app.cell
def _(adata, mo, plt, sc):
    mo.md("## Initial Data Overview")

    # Display basic UMAP plots
    overview_fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: Supertype
    sc.pl.umap(
        adata,
        color='supertype',
        legend_loc='on data',
        frameon=False,
        sort_order=False,
        na_in_legend=False,
        ax=axes[0],
        show=False
    )
    axes[0].set_title("Cell Supertypes")

    # Plot 2: Supertype scANVI
    sc.pl.umap(
        adata,
        color='supertype_scANVI',
        frameon=False,
        sort_order=False,
        na_in_legend=False,
        ax=axes[1],
        show=False
    )
    axes[1].set_title("Supertype scANVI")

    # Plot 3: Leiden clusters
    sc.pl.umap(
        adata,
        color='leiden',
        legend_loc='on data',
        frameon=False,
        sort_order=False,
        na_in_legend=False,
        ax=axes[2],
        show=False
    )
    axes[2].set_title("Leiden Clusters")

    plt.tight_layout()

    # Return the figure object as the last expression (Key component for Marimo functionality)
    overview_fig
    return


@app.cell
def _(mo, qc_vars_default):
    mo.md("## QC Metric Selection and Threshold Control")

    # QC metric selection
    qc_metric_selector = mo.ui.dropdown(
        options=qc_vars_default,
        value="Doublet score",
        label="Select QC Metric"
    )

    mo.vstack([
        mo.md("**Choose QC metric to analyze:**"),
        qc_metric_selector
    ])
    return (qc_metric_selector,)


@app.cell
def _(adata, mo, plot_QC_metric_hist, qc_metric_selector):
    # Display interactive distribution plot for selected QC metric
    dist_title = mo.md(f"### Distribution of {qc_metric_selector.value}")

    # Plot distribution for the selected metric
    if qc_metric_selector.value in adata.obs.columns:
        # Create seaborn/matplotlib histogram
        fig_hist = plot_QC_metric_hist(adata, qc_metric_selector.value)

        if fig_hist is not None:
            interactive_histogram = mo.mpl.interactive(fig_hist)
        else:
            interactive_histogram = mo.md("Could not create histogram")

        # Get some stats for threshold guidance
        cluster_means = adata.obs.groupby('leiden', observed=True)[qc_metric_selector.value].mean()
        overall_mean = adata.obs[qc_metric_selector.value].mean()
        overall_std = adata.obs[qc_metric_selector.value].std()

        stats_text = mo.md(f"""
        **Statistics for {qc_metric_selector.value}:**
        - Overall mean: {overall_mean:.4f}
        - Overall std: {overall_std:.4f}
        - Min cluster mean: {cluster_means.min():.4f}
        - Max cluster mean: {cluster_means.max():.4f}
        """)

        # Display everything together with interactive histogram
        display_output = mo.vstack([
            dist_title,
            interactive_histogram,
            stats_text
        ])
    else:
        display_output = mo.vstack([dist_title, mo.md(f"{qc_metric_selector.value} not found in data columns")])

    display_output
    return


@app.cell
def _(adata, mo, qc_metric_selector):
    # Dynamic threshold slider based on selected metric
    if qc_metric_selector.value in adata.obs.columns:
        metric_values = adata.obs[qc_metric_selector.value]
        metric_min = metric_values.min()
        metric_max = metric_values.max()

        # Set reasonable default thresholds based on metric type
        if qc_metric_selector.value == "Doublet score":
            default_threshold = 0.08
            slider_min = 0.0
            slider_max = min(0.5, metric_max)
            step = 0.005
        elif qc_metric_selector.value == "Genes detected":
            default_threshold = 1500
            slider_min = max(500, metric_min)
            slider_max = min(5000, metric_max)
            step = 50
        else:  # Fraction mitochondrial UMIs
            default_threshold = 0.020
            slider_min = 0.0
            slider_max = min(0.1, metric_max)
            step = 0.002

        threshold_slider = mo.ui.slider(
            start=slider_min,
            stop=slider_max,
            value=default_threshold,
            step=step,
            label=f"{qc_metric_selector.value} Threshold"
        )

    else:
        threshold_slider = mo.ui.slider(0, 1, 0.5, label="Threshold")

    return (threshold_slider,)


@app.cell
def _(adata, mo, qc_metric_selector, threshold_slider):
    # Display the threshold slider - ensure single last expression
    if qc_metric_selector.value in adata.obs.columns:
        display = mo.vstack([
            mo.md(f"**Adjust threshold for {qc_metric_selector.value}:**"),
            threshold_slider,
            mo.md(f"**Current threshold: {threshold_slider.value}**")
        ])
    else:
        display = mo.md("Selected metric not found in data")

    # This is the single last expression that gets displayed
    display
    return


@app.cell
def _(adata, create_umap_plots, mo, qc_metric_selector, threshold_slider):
    # Main QC visualization - updates when threshold changes
    mo.md("## QC Analysis Results")

    if qc_metric_selector.value in adata.obs.columns:
        # Determine direction based on metric type
        if qc_metric_selector.value == "Genes detected":
            qc_direction = "lt"  # less than threshold is bad
        else:
            qc_direction = "gt"  # greater than threshold is bad

        analysis_info_QC = mo.md(f"""
        **Current Analysis:**
        - Metric: {qc_metric_selector.value}
        - Threshold: {threshold_slider.value}
        - Direction: {'Less than' if qc_direction == 'lt' else 'Greater than'} threshold is flagged as poor quality
        """)

        # Create the plots
        qc_fig, adata_result, flagged_clusters = create_umap_plots(
            adata, 
            qc_metric_selector.value, 
            threshold_slider.value, 
            qc_direction
        )

    else:
        qc_fig = None
        adata_result = None
        flagged_clusters = []

    return adata_result, flagged_clusters, qc_fig


@app.cell
def _(
    adata,
    flagged_clusters,
    mo,
    qc_fig,
    qc_metric_selector,
    threshold_slider,
):
    # Display QC Analysis Results - ensure single last expression
    title_results = mo.md("## QC Analysis Results")

    if qc_metric_selector.value in adata.obs.columns and qc_fig is not None:
        # Determine direction based on metric type
        if qc_metric_selector.value == "Genes detected":
            direction = "lt"  # less than threshold is bad
        else:
            direction = "gt"  # greater than threshold is bad

        analysis_info = mo.md(f"""
        **Current Analysis:**
        - Metric: {qc_metric_selector.value}
        - Threshold: {threshold_slider.value}
        - Direction: {'Less than' if direction == 'lt' else 'Greater than'} threshold is flagged as poor quality
        """)

        clusters_info = mo.md(f"**Flagged clusters:** {', '.join(map(str, flagged_clusters)) if flagged_clusters else 'None'}")

        # Display everything together
        content = mo.vstack([title_results, analysis_info, qc_fig, clusters_info])

    else:
        content = mo.vstack([title_results, mo.md("Adjust the threshold slider above to see QC analysis plots")])

    # This is the single last expression that gets displayed
    content
    return


@app.cell
def _(adata_result, flagged_clusters, mo):
    # Summary statistics
    if adata_result is not None:
        # Calculate QC statistics
        flag_columns = [col for col in adata_result.obs.columns if col.endswith('_flag')]

        if flag_columns:
            flag_col = flag_columns[0]  # Take the first flag column
            total_cells = len(adata_result.obs)
            flagged_cells = (adata_result.obs[flag_col] == "True").sum()
            passed_cells = total_cells - flagged_cells

            mo.md(f"""
            ### QC Summary

            **Cell Counts:**
            - Total cells: {total_cells:,}
            - Cells flagged for removal: {flagged_cells:,} ({flagged_cells/total_cells*100:.1f}%)
            - Cells passing QC: {passed_cells:,} ({passed_cells/total_cells*100:.1f}%)

            **Cluster Analysis:**
            - Total clusters: {len(adata_result.obs['leiden'].unique())}
            - Flagged clusters: {len(flagged_clusters)}
            """)
        else:
            mo.md("No QC flags generated")
    else:
        mo.md("No analysis results available")
    return


if __name__ == "__main__":
    app.run()
