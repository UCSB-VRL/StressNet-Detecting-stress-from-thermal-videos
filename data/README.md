
# StressNet Dataset Release

We are pleased to announce the public release of the **StressNet thermal video dataset** used for the project:

**StressNet: Detecting Stress in Thermal Videos**

StressNet is a deep learning framework for estimating physiological signals and detecting stress states from thermal facial videos. The model uses thermal video recordings to reconstruct the **Initial Systolic Time Interval (ISTI)**, a physiological marker associated with cardiac sympathetic activity, and then classifies an individual’s stress state.

This dataset release is intended to support research in:

* Contactless physiological sensing
* Thermal video analysis
* Stress and affective-state recognition
* Biomedical computer vision
* Spatio-temporal deep learning
* Human-centered AI and behavioral health applications

## Dataset Access

The dataset is being released through the **BisQue image and data management platform** hosted at UCSB.

Dataset link:

https://bisque2.ece.ucsb.edu/client_service/view?resource=https://bisque2.ece.ucsb.edu/data_service/00-ZhxuAR4B9fqMrcD6pFFrLa

Researchers can use the BisQue interface to browse, inspect, and access the released StressNet dataset.

## About the Dataset

The StressNet dataset contains thermal video data collected for studying physiological responses and stress-state detection. The dataset was developed as part of the broader effort to evaluate whether thermal facial videos can be used to estimate physiological signals and classify stress in a non-contact manner.

The data supports the StressNet pipeline, including:

1. Thermal video input processing
2. Face-region thermal signal representation
3. ISTI signal estimation
4. Stress vs. no-stress classification

The dataset is released to encourage reproducibility, benchmarking, and further research in thermal-video-based physiological sensing.

## Repository Structure

This repository contains the source code and supporting files for the StressNet project.

```text
StressNet-Detecting-stress-from-thermal-videos/
│
├── data/
│   └── README.md              # Dataset release information
│
├── isti_predictor/            # ISTI signal prediction code
├── stress_predictor/          # Stress classification code
├── figures/                   # Project figures
├── requirements.txt           # Python dependencies
├── LICENSE
└── README.md
```

## Recommended Use

Researchers may use the dataset for academic and non-commercial research purposes, including:

* Reproducing StressNet experiments
* Developing new models for physiological signal estimation
* Benchmarking thermal-video-based stress-detection methods
* Exploring multimodal or temporal models for human-state recognition

Please ensure that any use of the dataset follows the dataset terms, institutional requirements, and any applicable human-subject research or privacy guidelines.

## Citation

If you use this dataset, code, or StressNet model in your research, please cite the following work:

```bibtex
@article{kumar2020stressnet,
  title={StressNet: Detecting Stress in Thermal Videos},
  author={Kumar, Satish and Iftekhar, A S M and Goebel, Michael and Bullock, Tom and MacLean, Mary H. and Miller, Michael B. and Santander, Tyler and Giesbrecht, Barry and Grafton, Scott T. and Manjunath, B. S.},
  journal={arXiv preprint arXiv:2011.09540},
  year={2020}
}
```

## Data Hosting

The dataset is hosted through **BisQue**, a scientific image informatics and data management platform developed at UCSB.

Dataset access:

https://bisque2.ece.ucsb.edu/client_service/view?resource=https://bisque2.ece.ucsb.edu/data_service/00-ZhxuAR4B9fqMrcD6pFFrLa

## License and Terms of Use

The source code in this repository is released under the repository license.

The dataset may have separate access, usage, or redistribution terms through BisQue. Users are responsible for reviewing and following all applicable dataset-use conditions before downloading, sharing, or publishing results derived from the data.

## Contact

For questions about the StressNet dataset or repository, please contact the project authors through the GitHub repository or cite the corresponding StressNet publication.

## Acknowledgment

We thank the contributors and collaborators involved in the development of the StressNet project and the UCSB BisQue platform for supporting the dataset release.
