from shapey.visualization.image_panel import DataPrepImagePanels, ImagePanelSingleSample
from shapey.visualization.correlation_fall_off import make_corr_fall_off_single_sample
import argparse
import os
import random
import numpy as np
from PIL import Image
import matplotlib

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
matplotlib.use('Agg')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='passes data directory and output file path')
    parser.add_argument('--input_dir', type=str, default=os.path.join(DATA_DIR, 'processed', 'your_feature.h5'))
    parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_DIR, 'figures', 'your_feature_figs'))
    parser.add_argument('--img_dir', type=str, default=os.path.join(PROJECT_DIR, 'data', 'ShapeY200', 'dataset'))

    args = parser.parse_args()

    print(args)

    input_name = args.input_dir
    output_dir = os.path.join(args.output_dir, 'image_panels') 
    os.makedirs(output_dir, exist_ok=True)

    image_panel_data_processor = DataPrepImagePanels(args.img_dir, args.input_dir)
    image_panel_single_sample = ImagePanelSingleSample(args.img_dir)

    # create randomly sampled image panel
    print('making randomly sampled image panels...')

    sampled_objs = np.random.choice(image_panel_data_processor.objnames, 20)
    exc_axes_of_interest = ['x','y','r','p','w','pr','rw','pw','prw']

    for obj in sampled_objs:
        for ax in exc_axes_of_interest:
            sampled_idx = random.randint(0,10)
            image_panel_single_sample.reset_figure()

            sampled_img, row1_info = image_panel_data_processor.get_best_matching_same_obj_with_exclusion(obj, ax, sampled_idx)
            _, row2_info = image_panel_data_processor.get_top11_best_matching_any_other_obj(obj, ax, sampled_idx)
            _, row3_info = image_panel_data_processor.get_top11_best_matching_per_any_other_obj(obj, ax, sampled_idx)
            _, row4_info = image_panel_data_processor.get_top11_best_matching_per_any_other_obj_category(obj, ax, sampled_idx)
            
            all_row_infos = [row1_info, row2_info, row3_info, row4_info]
            image_panel_single_sample.fill_top11_panel(sampled_img, all_row_infos)
            corrfalloff_fig = make_corr_fall_off_single_sample(all_row_infos, ax)
            
            image_panel_single_sample.figure.savefig(os.path.join(output_dir, obj + '-' + ax + str(sampled_idx+1) + '.png'))
            corrfalloff_fig.savefig(os.path.join(output_dir, obj + '-' + ax + str(sampled_idx+1) + '_corrfall.png'))
            
            #concatenate images
            im1 = Image.open(os.path.join(output_dir, obj + '-' + ax + str(sampled_idx+1) + '.png'))
            im2 = Image.open(os.path.join(output_dir, obj + '-' + ax + str(sampled_idx+1) + '_corrfall.png'))
            imnew = Image.new('RGB', (im1.width+im2.width, im1.height))
            imnew.paste(im1, (0,0))
            imnew.paste(im2, (im1.width, 0))
            imnew.save(os.path.join(output_dir, obj + '-' + ax + str(sampled_idx+1) + '.png'))
            os.remove(os.path.join(output_dir, obj + '-' + ax + str(sampled_idx+1) + '_corrfall.png'))
            matplotlib.pyplot.close(corrfalloff_fig)