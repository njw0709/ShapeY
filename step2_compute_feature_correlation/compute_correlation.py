from shapey.dataprocess.raw_data import compute_correlation_and_save
from shapey.utils.customdataset import HDFDataset, PermutationPairsDataset
from shapey.utils.configs import ShapeYConfig
import h5py

from hydra import compose, initialize
import logging
import traceback

log = logging.getLogger(__name__)

def compute_feature_correlation(args: ShapeYConfig) -> bool:
    hdfname = args.pipeline.step2_input
    feature_group_key = '/feature_output'
    completed = False
    try:
        hdfstore = h5py.File(hdfname, 'r+')
        feature_output_key = feature_group_key + '/output'
        if args.data.cr:
            feature_output_key_cr = feature_group_key + '/output_cr'

        print('Retrieving saved features...')
        original_features = hdfstore[feature_output_key]
        if args.data.cr:
            cr_features = hdfstore[feature_output_key_cr]
        num_features = len(original_features)

        # Now compute correlations between original / postprocessed image pairs
        log.info('Computing correlations...')
        corrval_key = '/pairwise_correlation'
        try:
            hdfstore.create_group(corrval_key)
        except ValueError:
            log.info(corrval_key + " already exists")
        
        if args.data.cr:
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
            log.info(corrval_key_original + " already exists")

        compute_correlation_and_save(permutation_dataset, hdfstore, corrval_key_original, batch_size=args.network.batch_size, num_workers=args.network.num_workers)
        completed = True
    except Exception as e:
        log.error(e)
        log.error(traceback.format_exc())
    finally:
        hdfstore.close()
        return completed


if __name__ == '__main__':
    with initialize(config_path="../conf", job_name="step2_compute_correlation"):
        cfg = compose(config_name="config", overrides=["data.project_dir=/home/namj/ShapeY"])
        compute_feature_correlation(cfg)

