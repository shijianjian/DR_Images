# Diabetic Retinopathy Images

Forgot to upload previous works and it all gone due to system errors. I am not gonna lose it again.

## Requirements
- Anaconda Python 3.6
- Tensorflow
- OpenCV

## Usage

### Data Preparation
Those data were from Kaggle Dataset,[Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection). This project provides some sample data comes from the source. Well, download the rest by yourself. It is around 90 gigabytes in total. Quiet a lot.

Please KEEP the folder structure like it is. 
- Put your new data into /data/images 
- Put the labels into /data/labels

#### Convert to TFRecords
Simple run
```bash
python data_preparation.py
```

TODO: add train/test separation later on.
