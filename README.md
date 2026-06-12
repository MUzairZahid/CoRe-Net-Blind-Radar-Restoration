# CoRe-Net: Co-Operational Regressor Network for Blind Radar Signal Restoration

This repository contains the main implementation of **CoRe-Net**, a co-operational learning framework for blind radar signal restoration. CoRe-Net uses an Apprentice Regressor (AR) and a Master Regressor (MR) to replace adversarial training with cooperative quality-regression feedback.

## Paper

**CoRe-Net: Co-Operational Regressor Network with Progressive Transfer Learning for Blind Radar Signal Restoration**

## Main Code

The main training and evaluation entry point is:

```bash
main.py
```

Both training and evaluation are controlled through the `--mode` argument:

```bash
python main.py --mode train
```

```bash
python main.py --mode evaluate
```

```bash
python main.py --mode both
```

Example command:

```bash
python main.py --mode both --dataset extended --device cuda --Eps 1.0 --Beta 10.0 --Phi 1.0 --psnr_pred_loss mse --scheduler cosine
```

The main script handles dataset loading, model initialization, training, checkpoint saving, and evaluation.

## Dataset

The BRSR benchmark dataset used in this study was introduced in the previous **BRSR-OpGAN** work and is available in a separate repository:

**BRSR-OpGAN Dataset Repository:**  
https://github.com/MUzairZahid/BRSR-OpGAN

Download the dataset from the original BRSR-OpGAN repository and place it in the expected dataset folder:

```bash
Prepared_Dataset/
```

The default dataset files expected by `main.py` are:

```bash
Prepared_Dataset/base_dataset_k0.pickle
Prepared_Dataset/extended_dataset_k0.pickle
```

## Citation

If you use this code, please cite the CoRe-Net paper. If you use the BRSR dataset, please also cite the original BRSR-OpGAN paper.

```bibtex
@misc{zahid2025corenet,
  title        = {CoRe-Net: Co-Operational Regressor Network with Progressive Transfer Learning for Blind Radar Signal Restoration},
  author       = {Zahid, Muhammad Uzair and Kiranyaz, Serkan and Yildirim, Alper and Gabbouj, Moncef},
  year         = {2025},
  eprint       = {2501.17125},
  archivePrefix= {arXiv},
  primaryClass = {eess.SP},
  url          = {https://arxiv.org/abs/2501.17125}
}
```

```bibtex
@article{zahid2025brsr,
  title   = {BRSR-OpGAN: Blind Radar Signal Restoration using Operational Generative Adversarial Network},
  author  = {Zahid, Muhammad Uzair and Kiranyaz, Serkan and Yildirim, Alper and Gabbouj, Moncef},
  journal = {Neural Networks},
  volume  = {190},
  pages   = {107709},
  year    = {2025},
  doi     = {https://doi.org/10.1016/j.neunet.2025.107709}
}
```

## License

This repository is released for academic and research use.
