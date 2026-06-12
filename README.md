````markdown
# CoRe-Net: Co-Operational Regressor Network for Blind Radar Signal Restoration

This repository contains the main implementation of **CoRe-Net**, a co-operational learning framework for blind radar signal restoration. CoRe-Net uses an Apprentice Regressor (AR) and a Master Regressor (MR) to replace adversarial training with cooperative quality-regression feedback.

## Paper

**CoRe-Net: Co-Operational Regressor Network with Progressive Transfer Learning for Blind Radar Signal Restoration**

## Main Code

The main training and evaluation entry point is:

```bash
main.py
````

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

Download the dataset from the original BRSR-OpGAN repository and place it in the expected dataset folder, for example:

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
@article{UzairZahid2025,
   abstract = {Real-world radar signals are frequently corrupted by various artifacts, including sensor noise, echoes, interference, and intentional jamming, differing in type, severity, and duration. This pilot study introduces a novel model, called Co-Operational Regressor Network (CoRe-Net) for blind radar signal restoration, designed to address such limitations and drawbacks. CoRe-Net replaces adversarial training with a novel cooperative learning strategy, leveraging the complementary roles of its Apprentice Regressor (AR) and Master Regressor (MR). The AR restores radar signals corrupted by various artifacts, while the MR evaluates the quality of the restoration and provides immediate and task-specific feedback, ensuring stable and efficient learning. The AR, therefore, has the advantage of both self-learning and assistive learning by the MR. The proposed model has been extensively evaluated over the benchmark Blind Radar Signal Restoration (BRSR) dataset, which simulates diverse real-world artifact scenarios. Under the fair experimental setup, this study shows that the CoRe-Net surpasses the Op-GANs over a 1 dB mean SNR improvement. To further boost the performance gain, this study proposes multi-pass restoration by cascaded CoRe-Nets trained with a novel paradigm called Progressive Transfer Learning (PTL), which enables iterative refinement, thus achieving an additional 2 dB mean SNR enhancement. Multi-pass CoRe-Net training by PTL consistently yields incremental performance improvements through successive restoration passes whilst highlighting CoRe-Net ability to handle such a complex and varying blend of artifacts.},
   author = {Muhammad Uzair Zahid and Serkan Kiranyaz and Senior Member and Alper Yildirim and Moncef Gabbouj},
   keywords = {Co-Operational Regressor Network,Index Terms-Blind radar signal restoration,Operational GANs,cooperative learning,progressive transfer learning},
   month = {1},
   title = {CoRe-Net: Co-Operational Regressor Network with Progressive Transfer Learning for Blind Radar Signal Restoration},
   url = {https://arxiv.org/pdf/2501.17125},
   year = {2025}
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
  doi     = {10.1016/j.neunet.2025.107709}
}
```

## License

This repository is released for academic and research use.

```
```
