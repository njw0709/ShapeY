import os
from shapey.visualization.image_panel import DataPrepImagePanels, ImagePanelErrorDisplay
import matplotlib
import argparse
from tqdm import tqdm

matplotlib.use('Agg')

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='passes data directory and output file path')
    parser.add_argument('--input_dir', type=str, default=os.path.join(DATA_DIR, 'processed', 'your_feature.h5'))
    parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_DIR, 'figures', 'your_feature_figs', 'error_panels'))
    parser.add_argument('--img_dir', type=str, default=os.path.join(PROJECT_DIR, 'data', 'ShapeY200', 'dataset'))

    args = parser.parse_args()

    print(args)

    input_name = args.input_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    image_panel_data_processor = DataPrepImagePanels(args.img_dir, args.input_dir)

    # create randomly sampled image panel
    print('making randomly sampled image panels...')

    exc_axes_of_interest = ['p','w','x','y','pr','rw','pw','prw']
    exc_dists = [3, 5, 7]

    for ax in tqdm(exc_axes_of_interest):
        for dist in exc_dists:
            if ax == 'p' and dist < 6:
                pass
            else:
                list_of_errors, cval_list = image_panel_data_processor.get_whole_error_list(ax, dist)
                for i in range(int(len(list_of_errors)/10)):
                    image_panel_error_display = ImagePanelErrorDisplay(args.img_dir)
                    image_panel_error_display.reset_figure()
                    ten_row_error = list_of_errors[i*10:(i+1)*10]
                    ten_row_cval = cval_list[i*10:(i+1)*10]
                    image_panel_error_display.fill_error_panel(ten_row_error, ten_row_cval, ax, dist, annotate=False)
                    image_panel_error_display.figure.savefig(os.path.join(output_dir, 'error_panel_{}_{}_{}.png'.format(ax, dist, i)))
                    
                    image_panel_error_display.reset_figure()
                    image_panel_error_display.fill_error_panel(ten_row_error, ten_row_cval, ax, dist, annotate=True)
                    image_panel_error_display.figure.savefig(os.path.join(output_dir + 'error_panel_{}_{}_{}_ann.png'.format(ax, dist, i)))
                    del image_panel_error_display
                    matplotlib.pyplot.close('all')
