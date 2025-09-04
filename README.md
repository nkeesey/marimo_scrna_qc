# Single-Cell QC Analysis Tool

This is an interactive Marimo application for quality control analysis of single-cell RNA-seq data, more information can be found on the Marimo docs: https://docs.marimo.io/

## Features

- **Interactive QC Analysis**: Select QC metrics, adjust thresholds, and visualize results on UMAP
- **Real-time Visualization**: Interactive plots that update as you adjust parameters
- **Multiple QC Metrics**: Support for Doublet score, Fraction mitochondrial UMIs, and Genes detected

## Getting started

Install the required dependencies:

```bash
conda activate {virtual env}
pip install scikit-misc igraph leidenalg marimo umap
```

## Usage

### Running the Application

```bash
marimo run qc.py # This starts application as an app
marimo edit qc.py # This starts application as editable notebook in Marimo UI
```

This will start the Marimo web application in the nearest browser tab. 

### QC Analysis Workflow

1. **Generate Data**: Load in your **preprocessed** scANVI data for your cell type of interest 
2. **Review Preprocessing**: Check that all required preprocessing steps are complete
3. **Select QC Metric**: Choose from available QC metrics (Doublet score, Fraction mitochondrial UMIs, Genes detected)
4. **Adjust Threshold**: Use the interactive slider to set QC thresholds
5. **Visualize Results**: View UMAP plots showing QC flagging results

## Troubleshooting

If you encounter import errors, ensure all dependencies are installed and no dependency conflicts exist with Marimo. 

```bash
pip install scikit-misc igraph leidenalg scanpy marimo matplotlib seaborn pandas numpy
```
