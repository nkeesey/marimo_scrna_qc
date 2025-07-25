import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def create_toy_dataset(
    n_cells: int = 200,
    n_genes: int = 1000,
    subclass: str = "Astrocyte",
    random_state: int | None = 0,
) -> sc.AnnData:
    """Generate a minimal synthetic ``AnnData`` object that mimics the end-product
    used in the QC/visualisation pipeline

    Parameters
    n_cells
        Number of synthetic cells to create.
    n_genes
        Number of synthetic genes to create.
    subclass
        Cell subclass label (e.g. *Astrocyte*). Appears in several metadata
        fields and file paths in the original pipeline.
    random_state
        Seed for NumPy random generator to get reproducible results.

    Returns
    sc.AnnData
        An ``AnnData`` object with:
        * expression matrix ``X`` (sparse CSR)
        * QC metrics columns: ``Genes detected``,
          ``Fraction mitochondrial UMIs``, ``Doublet score``
        * categorical metadata columns: ``supertype``, ``supertype_scANVI``,
          ``ADNC``, ``Donor ID``, ``Source``
        * pre-computed neighbour graph, Leiden clusters, UMAP coordinates
        * a fake scVI latent representation stored in ``obsm['X_scVI']`` so
          that the notebook can call ``use_rep='X_scVI'`` later on.
    """

    rng = np.random.default_rng(random_state)

    # 1. Simulate a counts matrix with a small fraction of mitochondrial genes
    gene_names = np.array([f"Gene{i}" for i in range(n_genes)], dtype=object)
    n_mt = max(5, int(0.02 * n_genes))  # ~2 % mitochondrial genes
    mt_indices = rng.choice(n_genes, n_mt, replace=False)
    gene_names[mt_indices] = [f"MT-{gene_names[i]}" for i in mt_indices]

    X = rng.poisson(lam=1.0, size=(n_cells, n_genes)).astype(np.int32)
    adata = sc.AnnData(sparse.csr_matrix(X))
    adata.var_names = gene_names

    # 2. QC metrics (match names expected downstream)
    n_genes_by_cell = (X > 0).sum(axis=1)
    total_counts = X.sum(axis=1)
    mito_counts = X[:, mt_indices].sum(axis=1)

    adata.obs["Genes detected"] = n_genes_by_cell
    adata.obs["Fraction mitochondrial UMIs"] = mito_counts / np.maximum(total_counts, 1)
    adata.obs["Doublet score"] = rng.uniform(0.0, 0.15, size=n_cells)

    # 3. Additional metadata columns referenced by the notebook
    adata.obs["Donor ID"] = rng.choice([f"Donor{i}" for i in range(1, 6)], size=n_cells)
    adata.obs["supertype"] = rng.choice([f"{subclass}_TypeA", f"{subclass}_TypeB"], size=n_cells)
    # Initially, let scANVI labels mirror the ground-truth labels
    adata.obs["supertype_scANVI"] = adata.obs["supertype"].copy()
    adata.obs["ADNC"] = rng.choice(["None", "Low", "High"], size=n_cells)
    adata.obs["Source"] = rng.choice(["DatasetA", "DatasetB"], size=n_cells)

    # 4. Minimal preprocessing to obtain PCA, neighbours, Leiden clusters & UMAP
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=200, subset=True, flavor="seurat_v3")
    sc.pp.pca(adata, n_comps=30, svd_solver="arpack", random_state=random_state)
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=30)
    sc.tl.leiden(adata, resolution=1.0, key_added="leiden", random_state=random_state)
    sc.tl.umap(adata, random_state=random_state)

    # 5. Provide a fake scVI latent space so that downstream calls using use_rep='X_scVI' run without modification
    adata.obsm["X_scVI"] = rng.normal(loc=0.0, scale=1.0, size=(n_cells, 20))

    adata.raw = adata.copy()

    return adata


def create_clustered_toy_dataset(
    n_cells: int = 500,
    n_genes: int = 2000,
    subclass: str = "Astrocyte",
    n_supertypes: int = 3,
    markers_per_cluster: int = 50,
    cluster_separation: float = 6.0,
    random_state: int | None = 0,
    latent_dim: int = 20,
) -> sc.AnnData:
    """Generate a synthetic ``AnnData`` object where each *supertype* forms a
    well-separated cluster in a latent space.  Useful for testing clustering and
    QC steps that should recover those supertypes
    """

    rng = np.random.default_rng(random_state)

    # 1. Gene names & mitochondrial genes
    gene_names = np.array([f"Gene{i}" for i in range(n_genes)], dtype=object)
    n_mt = max(5, int(0.02 * n_genes))
    mt_indices = rng.choice(n_genes, n_mt, replace=False)
    gene_names[mt_indices] = [f"MT-{gene_names[i]}" for i in mt_indices]
    non_mt_indices = np.setdiff1d(np.arange(n_genes), mt_indices)

    # 2. Allocate cells to clusters
    # Draw proportions from a Dirichlet then convert to integer sizes
    weights = rng.random(n_supertypes)
    weights /= weights.sum()
    cluster_sizes = (weights * n_cells).astype(int)
    # Ensure total matches exactly n_cells
    diff = n_cells - cluster_sizes.sum()
    if diff > 0:
        cluster_sizes[:diff] += 1
    elif diff < 0:
        # unlikely, but adjust
        cluster_sizes[:abs(diff)] -= 1

    # 3. Simulate latent space (scVI) using Gaussians
    latent_means = rng.normal(scale=cluster_separation, size=(n_supertypes, latent_dim))
    latent_data = np.zeros((n_cells, latent_dim), dtype=float)
    supertype_cluster_labels = np.empty(n_cells, dtype=int)

    # Cluster-specific marker genes
    marker_idx_per_cluster: list[list[int]] = []
    available_marker_pool = non_mt_indices.copy()
    rng.shuffle(available_marker_pool)

    for k in range(n_supertypes):
        start = np.sum(cluster_sizes[:k])
        end = start + cluster_sizes[k]
        # Latent vectors for this cluster
        latent_data[start:end, :] = rng.normal(
            loc=latent_means[k], scale=1.0, size=(cluster_sizes[k], latent_dim)
        )
        supertype_cluster_labels[start:end] = k

        # Select marker genes (allow reuse if pool depleted)
        if len(available_marker_pool) >= markers_per_cluster:
            marker_idx = available_marker_pool[:markers_per_cluster]
            available_marker_pool = available_marker_pool[markers_per_cluster:]
        else:
            marker_idx = rng.choice(non_mt_indices, size=markers_per_cluster, replace=False)
        marker_idx_per_cluster.append(marker_idx)

    # 4. Construct counts matrix with cluster-specific signal
    baseline_lambda = rng.uniform(0.5, 1.5, size=n_genes)
    X = np.zeros((n_cells, n_genes), dtype=np.int32)

    for i in range(n_cells):
        clu = supertype_cluster_labels[i]
        lam = baseline_lambda.copy()
        # Up-regulate marker genes for this cluster
        lam[marker_idx_per_cluster[clu]] += rng.uniform(2.0, 4.0)
        X[i, :] = rng.poisson(lam).astype(np.int32)

    adata = sc.AnnData(sparse.csr_matrix(X))
    adata.var_names = gene_names

    # 5. QC metrics
    n_genes_by_cell = (X > 0).sum(axis=1)
    total_counts = X.sum(axis=1)
    mito_counts = X[:, mt_indices].sum(axis=1)

    adata.obs["Genes detected"] = n_genes_by_cell
    adata.obs["Fraction mitochondrial UMIs"] = mito_counts / np.maximum(total_counts, 1)
    adata.obs["Doublet score"] = rng.uniform(0.0, 0.15, size=n_cells)

    # 6. Metadata
    supertype_names = [f"{subclass}_Type{chr(65 + k)}" for k in range(n_supertypes)]
    adata.obs["supertype"] = [supertype_names[k] for k in supertype_cluster_labels]
    adata.obs["supertype_scANVI"] = adata.obs["supertype"].copy()
    adata.obs["Donor ID"] = rng.choice([f"Donor{i}" for i in range(1, 6)], size=n_cells)
    adata.obs["ADNC"] = rng.choice(["None", "Low", "High"], size=n_cells)
    adata.obs["Source"] = rng.choice(["DatasetA", "DatasetB"], size=n_cells)

    # 7. Normalise & graph construction
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.obsm["X_scVI"] = latent_data  # store latent representation

    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=15)
    sc.tl.leiden(adata, resolution=1.0, random_state=random_state)
    sc.tl.umap(adata, min_dist=0.3, random_state=random_state)

    adata.raw = adata.copy()
    return adata


# Update public API
__all__ = [
    "create_toy_dataset",
    "create_clustered_toy_dataset",
]
