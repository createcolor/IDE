# Illumination Distribution Estimation

<img align="right" width="296" alt="Annotation 2023-11-06 201811" src="https://github.com/createcolor/IDE/assets/4645893/245b833c-6e27-44aa-bc6c-f2cec79bad49">

This is the official repository of

**Physically-Plausible Illumination Distribution Estimation**, Egor Ershov, Vasily Tesalin, Ivan Ermakov, Michael S. Brown. ICCV 2023. [Link](https://openaccess.thecvf.com/content/ICCV2023/html/Ershov_Physically-Plausible_Illumination_Distribution_Estimation_ICCV_2023_paper.html).

## Setup

### For Linux

```
python -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
```

### For Windows

```
python -m venv venv
./venv/bin/activate.bat
pip install -r requirements.txt
```

## Dataset

In this dataset we have provided new markup for mirror ball for [Cube++ dataset](https://zenodo.org/records/4153431). All markup is provided in markup folder [markup](https://github.com/createcolor/IDE/tree/develop/markup).

## Dataset generation

To generate GT distributions we need to use script [data_generator.py](https://github.com/createcolor/IDE/blob/develop/dataset_generators/dataset_generator.py) from folder [markup_generation](https://github.com/createcolor/IDE/tree/develop/dataset_generators).
This script is to generate
ball .PNG crops,
ball masks for generated .PNG crops,
ball markup w\ maximum level on the full image for every channel
using full image .PNGs and ball markup for cube crops, as well as
GT illumination distribution .PNG and .npy files for mirror ball crops
using crops .PNGs and masks.

The usage example with local paths:
```
python dataset_generators/dataset_generator.py --path_to_markup markup/ --path_to_save_cube3p ../data/cube_3p/ball_crops_png/ --path_to_cube2p ../Cube++/
```

## Solutions

#TODO
