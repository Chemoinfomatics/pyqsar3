# PyQsar3

## Description

Welcome to PyQsar3, an upgraded version of [pyqsar](https://github.com/crong-k/pyqsar_tutorial#readme), which originally worked only with Python 2.7. PyQsar3 is designed to work with Python 3.9 or higher, offering enhanced functionality and features.

## Installation

### Commands

- Install: `pip install .`
- Upgrade: `pip install . --upgrade`
- Reinstall: `pip uninstall pyqsar` and `pip install .`

### From GitHub




Firstly, Clone the repository and install:

```bash
git clone https://github.com/Chemoinfomatics/pyqsar3.git
cd pyqsar3
conda env create -f environment.yml 
#or use mama, which is way faster than conda
mamba env create -f environment.yml 

conda activate ssu-lab-pyqsar3
#or
mamba activate ssu-lab-pyqsar3 

pip install .
# or 
pip install pyqsar3

```

### Usage
For detailed usage instructions, refer to the Pyqsar3R.ipynb file included in this package.

### Citation
If you use PyQsar3 in your work, please cite the following paper:

Sinyoug Kim and Kwang-Hwi Cho, [PyQSAR: A Fast QSAR Modeling Platform Using Machine Learning and Jupyter Notebook. Bulletin of the Korean Chemical Society, 2019, 40.1: 39-44](https://onlinelibrary.wiley.com/doi/abs/10.1002/bkcs.11638)

### Questions
For any further questions or inquiries, feel free to contact us at chokh@ssu.ac.kr.
