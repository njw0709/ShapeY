from shapey.visualization.exclusion_distance import NNClassificationErrorV2
from shapey.utils.macroutils import make_axis_of_interest
import os
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')

from hydra import compose, initialize
import logging
from shapey.utils.configs import ShapeYConfig

log = logging.getLogger(__name__)

def graph_exclusion_top1(args: ShapeYConfig) -> bool:
    log.info("Generating Exclusion graphs...")
    input_name = args.pipeline.step4_input
    output_dir = os.path.join(args.pipeline.step4_output, 'exclusion_distance')
    os.makedirs(output_dir, exist_ok=True)
    common_fig_name = 'top1_error_'
    if args.graph.match_mode == 'category':
        common_fig_name += 'category_'
    if args.data.cr:
        common_fig_name += 'cr_{}_'.format(args.graph.cr_mode)
        input_name = input_name.split('.')[0] + '_cr.h5'
    axes = make_axis_of_interest()

    ones = [idx for idx, e in enumerate(axes) if len(e)==1]
    twos = [idx for idx, e in enumerate(axes) if len(e)==2][0:-3]
    threes = [idx for idx, e in enumerate(axes) if len(e)==3][0:-3]

    try:
        with h5py.File(input_name, 'r') as hdfstore:
            log.info("Pulling data...")
            imgnames = hdfstore['/feature_output/imgname'][:].astype('U')
            objnames = np.unique(np.array([c.split('-')[0] for c in imgnames]))
            if args.data.cr:
                key_head = '/contrast_reversed/{}'.format(args.graph.cr_mode)
            else:
                key_head = '/original'
            within_category = (args.graph.match_mode == 'category')
            res = [NNClassificationErrorV2.generate_top1_error_data(hdfstore, objnames, ax, key_head=key_head, within_category_error=within_category) for ax in axes]
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
            log.info("Making graphs...")
            for j, idxs in enumerate(average_ax_idx):
                res_list = [res[i] for i in idxs]
                ax_list = [axes[i] for i in idxs]
                res_unzipped = list(zip(*res_list))
                fig, fig_ax = NNClassificationErrorV2.plot_top1_err_avgd(res_unzipped[2], res_unzipped[3], ax_list)
                fig.set_size_inches(5.0, 6.5)
                fig.savefig(os.path.join(output_dir, common_fig_name + names[j] + '.png'), bbox_inches='tight', dpi=1200)
    except Exception as e:
        log.error("Failed to generate exclusion graphs: {}".format(e))
        return False
    log.info('done!')
    return True

if __name__ == '__main__':
    with initialize(config_path="../conf", job_name="step4_graph_exclusion_top1"):
        cfg = compose(config_name="config", overrides=["data.project_dir=/home/namj/ShapeY", 'network=resnet50_random', 'data=ShapeY200', 'graph=cat'])
        graph_exclusion_top1(cfg)