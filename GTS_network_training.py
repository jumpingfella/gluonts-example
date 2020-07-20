import pyximport; pyximport.install()
import sys
import pandas as pd
from GTS_new_data import load_dataset
from pathlib import Path
from gluonts.model.predictor import Predictor
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset

if __name__ == "__main__":
    filename = sys.argv[1]
    df = load_dataset(filename)

    training_data = ListDataset(
        [{"start": df.index[1], "target": df.iloc[:-12].values[:, 1]}],
        freq = "1min"
    )

    estimator = DeepAREstimator(freq="1min", prediction_length=12, trainer=Trainer(epochs=100))
    predictor = estimator.train(training_data=training_data)
    predictor.serialize(Path("."))

