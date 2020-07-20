import pandas as pd
import matplotlib.pyplot as plt
from gluonts.dataset.common import ListDataset
from itertools import islice
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer

def plot_forecasts(tss, forecasts, past_length, num_plots):
    for target, forecast in islice(zip(tss, forecasts), num_plots):
        ax = target[-past_length:].plot(figsize=(12, 5), linewidth=2)
        forecast.plot(color='g')
        plt.grid(which='both')
        plt.legend(["observations", "median prediction", "90% confidence interval", "50% confidence interval"])
        plt.show()

if __name__ == "__main__":

    url = "https://raw.githubusercontent.com/numenta/NAB/master/data/realTweets/Twitter_volume_AMZN.csv"
    df = pd.read_csv(url, header=0, index_col=0, parse_dates=True)

    training_data = ListDataset(
        [{"start": df.index[0], "target": df.value[:"2015-04-05 00:00:00"]}],
        freq = "5min"
    )

    estimator = DeepAREstimator(freq="5min",
                                prediction_length=36,
                                trainer=Trainer(epochs=1))

    predictor = estimator.train(training_data=training_data)

    test_data = ListDataset(
        [
            {"start": df.index[0], "target": df.value[:"2015-04-10 03:00:00"]},
            {"start": df.index[0], "target": df.value[:"2015-04-15 18:00:00"]},
            {"start": df.index[0], "target": df.value[:"2015-04-20 12:00:00"]}
        ],
        freq = "5min"
    )

    forecast_it, ts_it = make_evaluation_predictions(test_data, predictor=predictor, num_samples=100)
    forecasts = list(forecast_it)
    tss = list(ts_it)
    plot_forecasts(tss, forecasts, past_length=150, num_plots=3)
