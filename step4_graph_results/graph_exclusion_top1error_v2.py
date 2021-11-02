from shapey.visualization.exclusion_distance import NNClassificationErrorV2
from shapey.utils.macroutils import make_axis_of_interest
import argparse
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='passes data directory and output file path')
    parser.add_argument('--input_dir', type=str, default=os.path.join(DATA_DIR, 'processed', 'your_feature.h5'))
    parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_DIR, 'figures', 'your_feature_figs'))
    parser.add_argument('--within_category_error', type=int, default=0)

    args = parser.parse_args()

    print(args)

    input_name = args.input_dir
    output_dir = os.path.join(args.output_dir, 'exclusion_distance')
    os.makedirs(output_dir, exist_ok=True)
    common_fig_name = 'top1_error_'
    if args.within_category_error:
        common_fig_name += 'category_'
    axes = make_axis_of_interest()

    ones = [idx for idx, e in enumerate(axes) if len(e)==1]
    twos = [idx for idx, e in enumerate(axes) if len(e)==2][0:-3]
    threes = [idx for idx, e in enumerate(axes) if len(e)==3][0:-3]

    with h5py.File(input_name, 'r') as hdfstore:
        imgnames = hdfstore['/feature_output/imgname'][:].astype('U')
        objnames = np.unique(np.array([c.split('-')[0] for c in imgnames]))
        key_head = '/original'
       
        res = [NNClassificationErrorV2.generate_top1_error_data(hdfstore, objnames, ax, key_head=key_head, within_category_error=args.within_category_error) for ax in axes]
        res = list(zip(*res)) #top1_error_per_obj, top1_error_mean, num_correct_allobj, total_count
        res.append(axes)
        res = list(zip(*res))
        # figs = [NNClassificationErrorV2.plot_top1_err_per_axis(r[0], r[1], objnames, r[4]) for r in res]
        # for i, f in enumerate(figs):
        #     f[0].set_size_inches(15,13)
        #     f[0].savefig(output_dir + common_fig_name + axes[i] + '.png', bbox_inches="tight", dpi=1200)
        
        # generate averaged figure
        average_ax_idx = [ones, twos, threes]
        names = ['ones', 'twos', 'threes']
        for j, idxs in enumerate(average_ax_idx):
            res_list = [res[i] for i in idxs]
            ax_list = [axes[i] for i in idxs]
            res_unzipped = list(zip(*res_list))
            fig, fig_ax = NNClassificationErrorV2.plot_top1_err_avgd(res_unzipped[2], res_unzipped[3], ax_list)
            fig.set_size_inches(5.0, 6.5)
            fig.savefig(os.path.join(output_dir, common_fig_name + names[j] + '.png'), bbox_inches='tight', dpi=1200)
    print('done!')
