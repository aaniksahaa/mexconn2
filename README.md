### MExConn: A Mechanistically Interpretable Multi-Expert Framework for Multi-Organelle Segmentation in Connectomics
This repository contains  the official codebase for our work **MExConn** for multi-organelle segmentation from electron microscopy (EM) images with Mechanistic Interpretability.

## Repository Structure

`train.py`
Main training script for multi-organelle segmentation.

`single_org_train.py`
Script for the complete code of training models only for a specific organelle, which we refer to as SINGLE ORGANELLE TRAINING in our paper.

`channel_ablation.py`
Code for mechanistic interpretability analysis, computing encoder channel importance.

`test.py`
Script for evaluating trained models on various metrics - VI, Dice, IoU, F1, Recall.

`data/`
Directory containing the representative dataset.


## Usage
1. Train Multi-Organelle Segmentation Model
```
python train.py --data_root data --domain drosophila-vnc --out_path ./models/drosophila-vnc/model.pth --patch_size 256 --stride 128 --batch_size 8 --epochs 20 --lr 1e-4 --device cuda
```
2. Single Organelle Training

Train separate models for individual organelles:
```
python single_org_train.py --data_root data --domain drosophila-vnc --organelle mitochondria --out_path ./models/drosophila/model_mito.pth
python single_org_train.py --data_root data --domain drosophila-vnc --organelle synapses --out_path ./models/drosophila/model_syn.pth
python single_org_train.py --data_root data --domain drosophila-vnc --organelle membranes --out_path ./models/drosophila/model_mem.pth
```
3. Mechanistic Interpretability Analysis

Compute encoder channel importance:
```
python channel_ablation.py --data_root data --domain drosophila-vnc --model_path ./models/drosophila/model.pth --batch_size 8 --top_k 100 --device cuda

```
Detailed annotations are provided in the code for each step of the interpretability analysis.

4. Evaluation

Evaluate a trained model and print all metrics (VI, Dice, IoU, F1, Recall):
```
python test.py --data_root data --domain drosophila-vnc --model_path ./models/drosophila/model.pth --batch_size 8 --dev
```
