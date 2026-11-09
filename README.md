# Polyp Segmentation with PyTorch

## Question
How does the amount of labeled training data affect binary polyp segmentation performance?

## Dataset
Kvasir-SEG: https://datasets.simula.no/kvasir-seg/  

## Experiment
- Baseline: implement or adapt a small U-Net.
- Train on 10%, 25%, 50%, and 100% of the training split.
- Keep validation and test sets fixed across fractions.
- Primary metrics: Dice and IoU.
- Save training curves, per-run metrics, and qualitative prediction grids.

## Data layout
After downloading and extracting Kvasir-SEG, set `DATA_DIR` to the directory containing `images/` and `masks/`. Filenames should match between these folders. If the archive has a different layout, update `discover_pairs()`.

## Run
Open `notebooks/kvasir_segmentation.ipynb` in Colab. The notebook contains the dataset setup, deterministic split, TODO model section, training loop, evaluation, and plots.

