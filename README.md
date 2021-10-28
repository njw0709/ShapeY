# ShapeY

ShapeY is a benchmark that tests a vision system's shape recognition capacity. ShapeY currently consists of ~68k images of 200 3D objects taken from ShapeNet. Note that this benchmark is not meant to be used as a training dataset, but rather serves to validate that the visual object recogntion / classification under inspection has developed a capacity to perform well on our benchmarking tasks, which are designed to be hard if the system does not understand shape. Find our paper at: https://openreview.net/pdf?id=nN5ZRwsVaxU

## Installing ShapeY
Requirements: Python 3, Cuda version 10.2 (prerequisite for cupy)

To install ShapeY, run the following command:
```
pip3 install shapey==0.0.2
```

## Step0: Download ShapeY200 dataset
Run `download.sh` to download the dataset. The script automatically unzips the images under `data/ShapeY200/`.

## Step1: Extract the embedding vectors from your own vision model using our dataset
Implement the function `your_feature_output_code` in `step1_save_feature/your_feature_extraction_code.py`. The function should take in the path to the dataset as input and return two lists - one for the image names and another for the corresponding embedding vectors taken from whatever system.

## Step2: Run macro.py
After you have implemented the function, run `macro.py` to generate the results.
`macro.py` will automatically run the following steps:
1. Compute correlation between all embedding vectors (using `step2_compute_feature_correlation/compute_correlation.py`)

2. Run benchmark analysis (using `step3_benchmark_analysis/get_nn_classification_error_with_exclusion_distance.py`)

3. Graph results (top1 object matching error, top1 category matching error, etc.)


