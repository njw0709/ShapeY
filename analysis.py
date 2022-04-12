import sys
import hydra
from hydra.core.config_store import ConfigStore
import logging
from shapey.utils.configs import ShapeYConfig
import os

sys.path.append("./step1_save_feature")
sys.path.append("./step2_compute_feature_correlation")
sys.path.append("./step3_benchmark_analysis")

from save_feature2h5py import save_feature
from compute_correlation import compute_feature_correlation
from get_nn_classification_error_with_exclusion_distance import get_nn_classification_error

log = logging.getLogger(__name__)
cs = ConfigStore.instance()
cs.store(name='defaultconf', node=ShapeYConfig, group='grp')

@hydra.main(config_path="./conf", config_name='config')
def analyze(cfg: ShapeYConfig) -> None:
    print(cfg)
    log.info('setting gpu device to {}...'.format(cfg.network.gpu_num))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.network.gpu_num)
    log.info('starting macro for network: {}.'.format(cfg.network.name))
    log.info('saving feature outputs...')
    step1_completed = save_feature(cfg)
    if step1_completed:
        log.info('done saving feature outputs.')
    else:
        log.error('failed to save feature outputs.')
        return
    
    log.info('computing feature correlation...')
    step2_completed = compute_feature_correlation(cfg)
    if step2_completed:
        log.info('done computing feature correlation.')
    else:
        log.error('failed to compute feature correlation.')
        return
    
    log.info('getting nn classification error...')
    step3_completed = get_nn_classification_error(cfg)
    if step3_completed:
        log.info('done getting nn classification error.')
    else:
        log.error('failed to get nn classification error.')
        return
    return step3_completed
    log.info('graph exclusion top1 error...')
    step4_completed = graph_exclusion_top1(cfg)
    if step4_completed:
        log.info('done graph exclusion top1 error.')
    else:
        log.error('failed to graph exclusion top1 error.')
        return
    log.info('done with {}!'.format(cfg.network.name))
    
if __name__=="__main__":
    analyze()