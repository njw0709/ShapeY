from shapey.dataprocess.raw_data import AllImgPairCorrelationData, PostProcessedAllImgPairCorrelationData
from shapey.utils.configs import ShapeYConfig
from hydra import compose, initialize
import logging
log = logging.getLogger(__name__)

def get_tuning_curve(args: ShapeYConfig) -> None:
    input_name = args.pipeline.step3_output
    log.info('computing tuning curves...')
    try:
        if not args.data.cr:
            resnet_output_allimgpairs = AllImgPairCorrelationData(input_name)
            resnet_output_allimgpairs.compute_tuning_curves()
        else:
            resnet_output_allimgpairs = PostProcessedAllImgPairCorrelationData(input_name)
            resnet_output_allimgpairs.compute_tuning_curves()
    except Exception as e:
        log.error(e)
    finally:
        log.info('done!')
        resnet_output_allimgpairs.hdfstore.close()

if __name__ == '__main__':
     with initialize(config_path="../conf", job_name="step3_analysis_tuning_curve"):
        cfg = compose(config_name="config", overrides=["data.project_dir=/home/namj/ShapeY"])
        get_tuning_curve(cfg)