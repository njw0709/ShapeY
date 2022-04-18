import sys
import hydra
from hydra.core.config_store import ConfigStore
import logging
from shapey.utils.configs import ShapeYConfig
import os

sys.path.append("./step3_benchmark_analysis")


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
    
    log.info('getting nn classification error...')
    step3_completed = get_nn_classification_error(cfg)
    if step3_completed:
        log.info('done getting nn classification error.')
    else:
        log.error('failed to get nn classification error.')
        return
    return step3_completed
   
    
if __name__=="__main__":
    analyze()