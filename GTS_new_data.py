import pyximport; pyximport.install()
import sys
import pandas as pd
from pathlib import Path
from gluonts.model.predictor import Predictor
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset
import talib
import numpy as np
import warnings
import scipy.signal as sc

warnings.filterwarnings("ignore")

def load_dataset(filename):
    dataset = pd.read_csv(filename, usecols = [0, 1, 5], header=0)
    dataset = dataset.dropna()
    dataset.columns = dataset.columns.to_series().apply(lambda x: x.strip())
    df = dataset
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    timestamp = df.pop("timestamp")
    df.insert(0, timestamp.name, timestamp)
    df.drop(columns=['Date', 'Time'], inplace=True, errors='ignore')
    dataset = df

    features_to_normalize = ['close']
    dataset[features_to_normalize] = dataset[features_to_normalize].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
    return dataset

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size/2):]

def estimated_autocorrelation(x):
    """
    http://stackoverflow.com/q/14297012/190597
    http://en.wikipedia.org/wiki/Autocorrelation#Estimation
    """
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    #assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

if __name__ == "__main__":
    filename = sys.argv[1]
    df = load_dataset(filename)

    test_data = ListDataset(
        [{"start": df.index[1], "target": df.values[-12:, 1]}],
        freq="1 min"
    )

    predictor = Predictor.deserialize(Path("."))

    for test_entry, forecast in zip(test_data, predictor.predict(test_data)):
        print("[", forecast.mean[0], end = " ")

    np.seterr(divide='ignore')

    tsf5 = talib.TSF(df['close'].values, timeperiod=5)
    diff5 = np.diff(tsf5) / np.diff(df['close'].values)
    diff5 = np.insert(diff5, 0, 1)
    diff5 = np.diff(diff5) / np.diff(df['close'].values)

    tsf15 = talib.TSF(df['close'].values, timeperiod=15)
    diff15 = np.diff(tsf15) / np.diff(df['close'].values)
    diff15 = np.insert(diff15, 0, 1)
    diff15 = np.diff(diff15) / np.diff(df['close'].values)

    roc10 = talib.ROC(df['close'].values, timeperiod=10)
    # for local maxima
    arr = forecast.mean
    localMax = np.where(arr == np.amax(arr))

    # for local minima
    localMin = np.where(arr == np.amin(arr))

    print(forecast.mean[1], diff5[-1], diff15[-1], ' '.join(map(str, forecast.mean)), roc10[-1], end="]")
