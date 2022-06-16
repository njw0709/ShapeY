from shapey.dataprocess.raw_data import extract_features_resnet50
from shapey.utils.macroutils import check_image_order
from shapey.utils.configs import ShapeYConfig
from your_feature_extraction_code import extract_features_torch_with_hooks, timm_get_model

import traceback
import os
import h5py
import numpy as np
from hydra import compose, initialize
import logging

log = logging.getLogger(__name__)

def save_feature(args: ShapeYConfig) -> bool:
    if args.data.cr:
        datadir = args.data.cr_data_dir
    else:
        datadir = args.data.data_dir
    hdfname = args.pipeline.step1_output

    file_exists = os.path.exists(hdfname)
    feature_group_key = '/feature_output'
    completed = False
    try:
        if not args.data.cr and (args.pipeline.step1_recompute or not file_exists):
            hdfstore = h5py.File(hdfname, 'w')
            hdfstore.create_group(feature_group_key)
        else:
            hdfstore = h5py.File(hdfname, 'r+')

        if args.data.cr:
            imgname_key = feature_group_key + '/imgname_cr'
            feature_output_key = feature_group_key + '/output_cr'
        else:
            imgname_key = feature_group_key + '/imgname'
            feature_output_key = feature_group_key + '/output'

        if args.pipeline.step1_recompute or not file_exists or args.data.cr:
            if args.network.name == 'resnet50':
                log.info('Running example code (resnet50)')
                log.info('Extracting features from images')
                original_stored_imgname, original_stored_feat = extract_features_resnet50(datadir)
                origianl_imgnames = hdfstore.create_dataset(imgname_key, data=np.array(original_stored_imgname).astype('S'))
                original_features = hdfstore.create_dataset(feature_output_key, data=original_stored_feat)
                log.info('Saved resnet feature outputs!')
            else:
                #### put your feature extraction code here!! #####
                ## TODO: implement a version where you extract and save batches, not the whole thing at once 
                model_name = args.network.name
                model = timm_get_model(model_name, args)
                l = [m for m in model.named_modules()]
                feature_layer = l[-1][0]
                log.info('Extracting features from images ({})'.format(model_name))
                original_stored_imgname, original_stored_feat = extract_features_torch_with_hooks(datadir, model, feature_layer, input_img_size=args.network.input_size)
                imgname_order = np.array(original_stored_imgname)
                imgname_order = imgname_order.astype('U')
                reference_imgname = np.load(os.path.join(args.data.project_dir, 'step1_save_feature', 'imgname_ref.npy'))
                assert check_image_order(original_stored_imgname, reference_imgname)
                try:
                    original_imgnames = hdfstore.create_dataset(imgname_key, data=np.array(original_stored_imgname).astype('S'))
                    original_features = hdfstore.create_dataset(feature_output_key, data=original_stored_feat)
                except Exception as e:
                    log.info('Error: {}'.format(e))
                    del hdfstore[imgname_key]
                    del hdfstore[feature_output_key]
                    original_imgnames = hdfstore.create_dataset(imgname_key, data=np.array(original_stored_imgname).astype('S'))
                    original_features = hdfstore.create_dataset(feature_output_key, data=original_stored_feat)
                log.info('Saved {} feature outputs!'.format(args.network.name))
        else:
            log.info('Retrieving saved features...')
            original_features = hdfstore[feature_output_key]
            assert len(original_features) == 68200
            log.info('Features already exist!')
        completed = True
    except Exception as e:
        log.error(e)
        log.error(traceback.format_exc())
    finally:
        hdfstore.close()
        return completed

if __name__ == '__main__':
     with initialize(config_path="../conf", job_name="step1_save_feature"):
        cfg = compose(config_name="config", overrides=["data.project_dir=/home/namj/ShapeY"])
        save_feature(cfg)

   