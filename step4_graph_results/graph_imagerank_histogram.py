from shapey.visualization.histogram import ImageRankHistogram
import argparse
import os
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.use('Agg')

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='passes data directory and output file path')
    parser.add_argument('--input_dir', type=str, default=os.path.join(DATA_DIR, 'processed', 'your_feature_objnum_exp.h5'))
    parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_DIR, 'figures', 'your_feature_figs', 'numexp'))
    parser.add_argument('--within_category_error', type=bool, default=False)

    args = parser.parse_args()

    print(args)

    input_name = args.input_dir
    output_dir = os.path.join(args.output_dir, 'histogram')
    os.makedirs(output_dir, exist_ok=True)
    common_fig_name = 'objrank_histogram_'
    exc_dist = 3
    colors = cm.get_cmap('magma', 20)

    with h5py.File(input_name, 'r') as hdfstore:
        axes = ['x','y','r','p','w','pr','rw','pw','prw']
        imgrank_histogram_drawer = ImageRankHistogram(hdfstore)
        for ax in axes:
            f, a = plt.subplots(1,1)
            #num obj on the same plot
            for i, num_obj in enumerate([20, 40, 80, 160, 200]):
                try:
                    histcounts, density_hist, bins = imgrank_histogram_drawer.get_objrank_histogram(hdfstore, ax, num_objs=num_obj)
                    a.plot(bins[:-1] + np.diff(bins), density_hist[exc_dist], linestyle='-', color=colors(i), alpha=0.8, linewidth=1.5, label=num_obj)
                    #exclusion distance on the same plot
                    f2, a2 = plt.subplots(1,1)
                    for dist in range(0, 11):
                        a2.plot(bins[:-1] + np.diff(bins), density_hist[dist], linestyle='-', color=colors(dist), alpha=0.8, linewidth=1.5, label='exc. dist = {}'.format(dist))
                    a2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
                    f2.savefig(os.path.join(output_dir, common_fig_name + ax + '_{}_all_excdists'.format(num_obj) +'.png'), bbox_inches="tight", dpi=400)
                    plt.close()
                except:
                    pass

            a.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
            f.savefig(os.path.join(output_dir, common_fig_name + ax + str(exc_dist) +'.png'), bbox_inches="tight", dpi=400)
        print('done!')
