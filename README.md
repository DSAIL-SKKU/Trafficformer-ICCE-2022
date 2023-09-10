# Trafficformer: A Transformer-based Traffic Predictor

We propose a deep learning model that can predict future traffic. The proposed Transformer-based model is evaluated using the visit data of popular Wikipedia pages for more than 2 years through multiple accessing devices such as mobile and desktop. The experiment results demonstrate that the proposed model can predict the future traffic with high accuracy.

This codebase contains the python scripts for the model for the ICCE 2022. https://ieeexplore.ieee.org/document/9730205

## Data

To evaluate the proposed model, we use the large-scale Wikipedia dataset consisting of the daily numbers of the visits of 145 K Wikipedia articles from July 1, 2015 to September 10, 2017, available on [Kaggle](https://www.kaggle.com/c/web-traffic-time-series-forecasting/data). In additional to the total numbers of daily visits of each page, the dataset also provides the daily numbers of the visits via 3 different types of accessing devices: Spider, Desktop, and Mobile.

## Run

```python
python train.py --filename --n_days --save name

## Example
python train.py train.csv 10 train
```

—filename: A factor that selects a data file for training.
—n_days: A factor that determines whether to view the next data based on information about n days.
—save_name: A factor that receives where to store the model.



```python
python predict.py --filename --save_name

## Example
python predict.py predict.csv predict
```

—filename: Data on which files to view and make predictions.
—save_name: A factor that puts the storage name of the trained model.

The figure below defines the Trafficformer architecture.
<img width="466" alt="image" src="https://github.com/DSAIL-SKKU/Trafficformer-ICCE-2022/assets/60170358/16a9e478-6fe3-4cb0-9436-3ab8059470fa">

## Performance

<img width="459" alt="image" src="https://github.com/DSAIL-SKKU/Trafficformer-ICCE-2022/assets/60170358/284143f9-b868-4034-a048-07eedfb88b7e">

All the MAPEs and RMSEs are low while the R2- scores tend to be high, meaning that the proposed model can predict the visit counts accurately. In particular, the proposed model outperforms the baseline across all accessing devices; the MAPE values of Desktop, Mobile, and Spider are 0.138, 0.061, and 0.851 in Trafficformer, respectively, while the ones of Desktop, Mobile, and Spider are 0.921, 0.690, and 4.994 in LSTM, respectively. The R2-scores of LSTM for three types of the devices are negative while the ones of the proposed model are higher than 0.5. This results show that the proposed model can be used for accurately predicting traffic for diverse devices. Note that the performance of the visit counts for mobile devices is higher than those of others, implying that the proposed model can estimate the amount of mobile traffic, e.g., in 5G/6G networks, more accurately.
