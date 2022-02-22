from shapey.dataprocess.raw_data import extract_features_resnet50, compute_correlation_and_save
from shapey.utils.customdataset import HDFDataset, PermutationPairsDataset
import torch
import argparse
import os
import h5py
import numpy as np

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='passes data directory and output hdf file name')
    parser.add_argument('--feature_name', type=str, default='your_feature_name')
    parser.add_argument('--input_dir', type=str, default=os.path.join(DATA_DIR, 'intermediate', 'your_feature.h5'))
    parser.add_argument('--contrast_reversed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=20000)
    parser.add_argument('--num_workers', type=int, default=0)

    args = parser.parse_args()

    print(args)

    hdfname = args.input_dir

    first_time = os.path.exists(args.input_dir)
    feature_group_key = '/feature_output'
    try:
        hdfstore = h5py.File(hdfname, 'r+')

        imgname_key = feature_group_key + '/imgname'
        feature_output_key = feature_group_key + '/output'
        if args.contrast_reversed:
            imgname_key_cr = feature_group_key + '/imgname_cr'
            feature_output_key_cr = feature_group_key + '/output_cr'

        print('Retrieving saved features...')
        original_features = hdfstore[feature_output_key]
        if args.contrast_reversed:
            cr_features = hdfstore[feature_output_key_cr]
        num_features = len(original_features)

        # Now compute correlations between original / postprocessed image pairs
        print("Computing correlations..")
        corrval_key = '/pairwise_correlation'
        try:
            hdfstore.create_group(corrval_key)
        except ValueError:
            print(corrval_key + " already exists")
        
        if args.contrast_reversed:
            mem_usage = 0.42
            original_dataset = HDFDataset(original_features, mem_usage=mem_usage)
            cr_dataset = HDFDataset(cr_features, mem_usage=mem_usage)
            permutation_dataset = PermutationPairsDataset(original_dataset, postprocessed=cr_dataset)
            corrval_key_original = corrval_key + '/contrast_reversed'
        else:
            mem_usage = 0.85
            original_dataset = HDFDataset(original_features, mem_usage=mem_usage)
            permutation_dataset = PermutationPairsDataset(original_dataset)
            corrval_key_original = corrval_key + '/original'

        try:
            hdfstore.create_dataset(corrval_key_original, shape=(num_features, num_features))
        except ValueError:
            print(corrval_key_original + " already exists")
        compute_correlation_and_save(permutation_dataset, hdfstore, corrval_key_original, batch_size=args.batch_size, num_workers=args.num_workers)
        
    finally:
        hdfstore.close()