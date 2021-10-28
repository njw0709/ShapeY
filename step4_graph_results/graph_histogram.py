from shapey.visualization.histogram import HistogramPlot
import argparse
import numpy as np
import pandas as pd
import os

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='passes data directory and output file path')
    parser.add_argument('--input_dir', type=str, default=os.path.join(DATA_DIR, 'processed', 'your_feature.h5'))
    parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_DIR, 'figures', 'your_feature_figs'))

    args = parser.parse_args()

    print(args)

    input_name = args.input_dir
    output_dir = os.path.join(args.output_dir, 'histogram')
    os.makedirs(output_dir, exist_ok=True)
    output_name = os.path.join(output_dir, 'normalized_correlation_value_histogram_same_different_obj_separated.png')

    hdfstore = pd.HDFStore(input_name, 'r')
    df_normalized = hdfstore['/pairwise_correlation/histogram_normalized/']

    xvals = np.array(df_normalized.index)
    labels = list(df_normalized.columns)
    hists = [df_normalized[l].values for l in labels]

    print('making histogram plots...')
    histplot = HistogramPlot(xvals, hists, labels)
    f = histplot.make_figure()
    f.write_image(output_name)
    print('done!')