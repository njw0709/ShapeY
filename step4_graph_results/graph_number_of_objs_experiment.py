from shapey.visualization.exclusion_distance import NNClassificationErrorV2
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
    parser.add_argument('--within_category_error', type=int, default=0)

    args = parser.parse_args()

    print(args)

    input_name = args.input_dir
    output_dir = os.path.join(args.output_dir, 'exclusion_distance')
    os.makedirs(output_dir, exist_ok=True)
    common_fig_name = 'top1_error_'

    with h5py.File(input_name, 'r') as hdfstore:
        imgnames = hdfstore['/feature_output/imgname'][:].astype('U')
        objnames_original = np.unique(np.array([c.split('-')[0] for c in imgnames]))
        axes = ['x','y','r','p','w','pr','rw','pw','prw']
        num_obj_res = []
        for num_obj in range(20, 210, 20):
            if not args.post_processed:
                key_head = '/original_{}'.format(num_obj)
            else:
                key_head = '/postprocessed_{}/{}'.format(num_obj, args.pp_mode)
            obj_idx = hdfstore['/pairwise_correlation' + key_head + '_obj_idx'][:]
            objnames = objnames_original[obj_idx]
            res = [NNClassificationErrorV2.generate_top1_error_data(hdfstore, objnames, ax, key_head=key_head, within_category_error=args.within_category_error) for ax in axes]
            res = list(zip(*res)) #top1_error_per_obj, top1_error_mean, num_correct_allobj, total_count
            res.append(axes)
            res = list(zip(*res))
            num_obj_res.append(res)

        num_obj_res = list(zip(*num_obj_res))
        mStyles = [".",",","v","^","<",
                ">","1","2","3","4",
                "8","s","p","P","*",
                "h","H","+","x","X"]
        colors = cm.get_cmap('magma', 20)
        num_objs = list(range(20, 210, 20))
        draw = [20, 40, 80, 160]
        for ax_res in num_obj_res:
            figs, axes = plt.subplots(1,1)
            for i, res in enumerate(ax_res):
                top1_error_per_obj = res[0]
                top1_error_mean = res[1]
                exc_ax = res[-1]
                all_data = np.array(top1_error_per_obj)
                if num_objs[i] in draw:
                    axes.errorbar(np.linspace(0,10,11), top1_error_mean.T, 
                                yerr=[top1_error_mean.T-np.quantile(all_data, 0.25,axis=0), np.quantile(all_data, 0.75,axis=0)-top1_error_mean.T],
                                linestyle='-', alpha=0.8, linewidth=1.5, color=colors(i) ,marker=mStyles[i], label=num_objs[i], capsize=2)
            axes.set_ylim([-0.1, 1.1])
            axes.set_xlim([-0.5, 10.5])
            major_ticks = np.arange(0, 11, 1)
            minor_ticks = np.arange(-0.5, 10.5, 1)
            axes.set_xticks(major_ticks, minor=False)
            axes.set_xticks(minor_ticks, minor=True)
            axes.set_yticks([-0.1, 1.1], minor=True)
            axes.grid(linestyle='--', alpha=0.5, which='major')
            axes.tick_params('x', length=0, which='major')
            axes.grid(linestyle='-', alpha=1, which='minor')
            for i in range(int(len(minor_ticks)/2)):
                axes.axvspan(minor_ticks[2*i+1], minor_ticks[2*(i+1)], facecolor='g', alpha=0.1)
            axes.set_xlabel('Exclusion Distance')
            axes.set_ylabel('Top1 Nearest Neighbor Classification Error')
            axes.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1)
            figs.savefig(os.path.join(output_dir, common_fig_name + exc_ax + '.png'), bbox_inches="tight", dpi=400)
        print('done!')
