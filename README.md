# StressNet: Detecting Stress in Thermal Videos

[![Python](https://img.shields.io/badge/python-3.5--3.8-brightgreen?style=flat&logo=python&color=green)]()
[![PyTorch](https://img.shields.io/badge/library-PyTorch-blue?style=flat&logo=pytorch&color=informational)]()
[![License](https://img.shields.io/cocoapods/l/AFNetworking)]()

Official repository for our **WACV 2021** paper:

### [StressNet: Detecting Stress in Thermal Videos](https://openaccess.thecvf.com/content/WACV2021/html/Kumar_StressNet_Detecting_Stress_in_Thermal_Videos_WACV_2021_paper.html)

[Satish Kumar*](https://www.linkedin.com/in/satish-kumar-81912540/), [ASM Iftekhar](https://www.linkedin.com/in/a-s-m-iftekhar-86914b136/), [Michael Goebel](https://www.linkedin.com/in/mike-goebel-6331551bb/), [Tom Bullock](https://www.linkedin.com/in/tomwbullock/), [Mary H. MacLean](https://psych.ucsb.edu/people/researchers/mary-maclean), [Michael B. Miller](https://psych.ucsb.edu/people/michael-miller), [Tyler Santander](https://psych.ucsb.edu/people/researchers/tyler-santander), [Barry Giesbrecht](https://psych.ucsb.edu/people/faculty/barry-giesbrecht), [Scott T. Grafton](https://psych.ucsb.edu/people/faculty/scott-grafton), [B.S. Manjunath](https://vision.ece.ucsb.edu/people/bs-manjunath)

[[Paper]](https://openaccess.thecvf.com/content/WACV2021/html/Kumar_StressNet_Detecting_Stress_in_Thermal_Videos_WACV_2021_paper.html)  
[[arXiv]](https://arxiv.org/pdf/2011.09540.pdf)  
[[Dataset]](https://bisque2.ece.ucsb.edu/client_service/view?resource=https://bisque2.ece.ucsb.edu/data_service/00-ZhxuAR4B9fqMrcD6pFFrLa)

---

## Dataset Release Announcement

We are pleased to announce that the **StressNet thermal video dataset** is now publicly released through the UCSB **BisQue** platform.

The dataset can be accessed here:

**[StressNet Dataset on BisQue](https://bisque2.ece.ucsb.edu/client_service/view?resource=https://bisque2.ece.ucsb.edu/data_service/00-ZhxuAR4B9fqMrcD6pFFrLa)**

This release is intended to support research in contactless physiological sensing, thermal video analysis, stress-state recognition, biomedical computer vision, and human-centered AI.

More details about the dataset release are available in:

```text
data/README.md
```

---

## Overview

StressNet introduces a fast and novel deep learning framework for estimating physiological signals and classifying stress states from thermal videos.

The method uses thermal facial videos to estimate the **Initial Systolic Time Interval (ISTI)**, a physiological signal associated with cardiac sympathetic activity. The predicted ISTI signal is then used to classify whether an individual is experiencing stress.

This repository provides a ground-up implementation of the major components of StressNet. The code is written in Python and powered by the PyTorch deep learning framework.

<p align="center">
  <img src="figures/overview.gif" width="700">
</p>

---

## Repository Contents

This repository includes:

- Source code for the ISTI signal predictor using a ResNet50-based baseline model
- Source code for the stress detector
- Training and testing code for both ISTI prediction and stress classification
- Requirements file for setting up the environment
- Example workflow for training on your own dataset
- Dataset release information through BisQue

---

## Repository Structure

```text
StressNet-Detecting-stress-from-thermal-videos/
│
├── data/
│   └── README.md              # Dataset release and access information
│
├── isti_predictor/            # ISTI signal prediction code
├── stress_predictor/          # Stress classification code
├── figures/                   # Figures and overview animation
├── requirements.txt           # Python dependencies
├── LICENSE
└── README.md                  # Main project README
```

---

## Requirements

- Linux or macOS
- Python >= 3.5
- PyTorch >= 1.4.0
- CUDA 10.1, 10.0, or 9.2
- cuDNN compatible with the installed CUDA version

---

## Installation

Clone the repository:

```bash
git clone https://github.com/UCSB-VRL/StressNet-Detecting-stress-from-thermal-videos.git
cd StressNet-Detecting-stress-from-thermal-videos
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ISTI Signal Predictor

The ISTI predictor estimates the ISTI signal from thermal video input. The output is a NumPy array containing the predicted ISTI signal.

The source code is available in:

```text
isti_predictor/
```

Please follow the module-specific instructions here:

[isti_predictor/README.md](https://github.com/UCSB-VRL/StressNet-Detecting-stress-from-thermal-videos/blob/main/isti_predictor/README.md)

---

## Stress Detector

To detect stress from a thermal video, first run the ISTI predictor. The predicted ISTI signal is then passed to the stress detector to estimate the probability that the individual is experiencing stress.

The source code is available in:

```text
stress_predictor/
```

Please follow the module-specific instructions here:

[stress_predictor/README.md](https://github.com/UCSB-VRL/StressNet-Detecting-stress-from-thermal-videos/blob/main/stress_predictor/README.md)

---

## Dataset

The StressNet dataset is hosted on the UCSB BisQue platform.

Access the dataset here:

**[StressNet Dataset on BisQue](https://bisque2.ece.ucsb.edu/client_service/view?resource=https://bisque2.ece.ucsb.edu/data_service/00-ZhxuAR4B9fqMrcD6pFFrLa)**

The dataset supports research on:

- Thermal video-based physiological sensing
- ISTI signal estimation
- Stress-state classification
- Spatio-temporal learning from thermal data
- Contactless biomedical AI systems

Please review the dataset access and usage information in:

```text
data/README.md
```

Users are responsible for following all applicable dataset-use conditions, institutional requirements, and privacy or human-subject research guidelines.

---

## Citation

If this work is useful to your research, please cite our paper:

```bibtex
@inproceedings{kumar2020stressnet,
  title={StressNet: Detecting Stress in Thermal Videos},
  author={Kumar, Satish and Iftekhar, ASM and Goebel, Michael and Bullock, Tom and MacLean, Mary H and Miller, Michael B and Santander, Tyler and Giesbrecht, Barry and Grafton, Scott T and Manjunath, BS},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={999--1009},
  year={2021}
}
```

---

## License

Please refer to the repository license for code usage terms.

The dataset may have separate access, usage, or redistribution terms through BisQue. Please review the dataset release information before using or redistributing the data.

---

## Acknowledgment

We thank all collaborators and contributors involved in the StressNet project, as well as the UCSB BisQue platform for supporting the dataset release.
