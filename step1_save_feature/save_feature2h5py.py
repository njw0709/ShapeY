from shapey.dataprocess.raw_data import extract_features_resnet50
from shapey.utils.macroutils import check_image_order
from your_feature_extraction_code import extract_features_torch_with_hooks, simclr_resnet, timm_get_model

import argparse
import os
import h5py
import numpy as np

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='passes data directory and output hdf file name')
    parser.add_argument('--input_dir', type=str, default=os.path.join(DATA_DIR, 'ShapeY200'))
    parser.add_argument('--output_dir', type=str, default=os.path.join(DATA_DIR, 'intermediate'))
    parser.add_argument('--contrast_reversed', type=int, default=0)
    parser.add_argument('--contrast_reversed_input_dir', type=str, default=os.path.join(DATA_DIR, 'ShapeY200CR'))
    parser.add_argument('--recompute_feat', type=int, default=0)
    parser.add_argument('--run_example', type=int, default=1)
    parser.add_argument('--name', type=str, default='ResNet50')
    parser.add_argument('--feature_layer', type=str, default='avgpool')
    parser.add_argument('--input_size', type=int, default=224)

    args = parser.parse_args()

    print(args)

    if args.contrast_reversed:
        datadir = args.contrast_reversed_input_dir
    else:
        datadir = args.input_dir
    hdfname = os.path.join(args.output_dir, args.name+'.h5')

    first_time = os.path.exists(hdfname)
    feature_group_key = '/feature_output'
    try:
        if not args.contrast_reversed and (args.recompute_feat or first_time):
            hdfstore = h5py.File(hdfname, 'w')
            hdfstore.create_group(feature_group_key)
        else:
            hdfstore = h5py.File(hdfname, 'r+')

        if args.contrast_reversed:
            imgname_key = feature_group_key + '/imgname_cr'
            feature_output_key = feature_group_key + '/output_cr'
        else:
            imgname_key = feature_group_key + '/imgname'
            feature_output_key = feature_group_key + '/output'

        if args.recompute_feat or first_time:
            if args.run_example:
                print('Extracting resnet feature outputs...')
                original_stored_imgname, original_stored_feat = extract_features_resnet50(datadir)
                origianl_imgnames = hdfstore.create_dataset(imgname_key, data=np.array(original_stored_imgname).astype('S'))
                original_features = hdfstore.create_dataset(feature_output_key, data=original_stored_feat)
                print('Saved resnet feature outputs!')
            else:
                #### put your feature extraction code here!! #####
                ## TODO: implement a version where you extract and save batches, not the whole thing at once 
                model_name = args.name
                model = timm_get_model(model_name)
                original_stored_imgname, original_stored_feat = extract_features_torch_with_hooks(datadir, model, args.feature_layer, input_img_size=args.input_size)
                imgname_order = np.array(original_stored_imgname)
                imgname_order = imgname_order.astype('U')
                reference_imgname = np.load(os.path.join(PROJECT_DIR, 'step1_save_feature', 'imgname_ref.npy'))
                assert check_image_order(original_stored_imgname, reference_imgname)
                origianl_imgnames = hdfstore.create_dataset(imgname_key, data=np.array(original_stored_imgname).astype('S'))
                original_features = hdfstore.create_dataset(feature_output_key, data=original_stored_feat)
                print('Saved {} feature outputs!'.format(args.name))
        else:
            print('Retrieving saved features...')
            original_features = hdfstore[feature_output_key]
            assert len(original_features) == 68200
            print('Features already exist!')
    finally:
        hdfstore.close()