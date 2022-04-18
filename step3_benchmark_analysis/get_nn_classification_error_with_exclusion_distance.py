from regex import E
from shapey.dataprocess.raw_data import ImgCorrelationDataProcessorV2
import shutil
import os
import h5py

from hydra import compose, initialize
import logging
from shapey.utils.configs import ShapeYConfig

log = logging.getLogger(__name__)

def get_nn_classification_error(args: ShapeYConfig) -> bool:
    input_name = args.pipeline.step3_input
    output_name = args.pipeline.step3_output

    if args.data.cr:
        output_name = output_name.split('.')[0] + '_cr.h5'

    # first check and copy the h5 file
    if not os.path.exists(output_name):
        log.info("Copying h5 file...")
        shutil.copyfile(input_name, output_name)
    else:
        log.info('Output file already exists. Removing and copying again...')
        os.remove(output_name)
        shutil.copyfile(input_name, output_name)
    completed = False
    try:
        with h5py.File(output_name, 'a') as hdfstore:
            data_processor = ImgCorrelationDataProcessorV2(hdfstore)
            if args.data.cr:
                data_processor.exclusion_distance_analysis(hdfstore, contrast_reversed=args.data.cr, exclusion_mode='soft')
                data_processor.exclusion_distance_analysis(hdfstore, contrast_reversed=args.data.cr, exclusion_mode='hard')
            else:
                data_processor.exclusion_distance_analysis(hdfstore)
            completed = True
    except Exception as e:
        log.error("Failed to do exclusion distance analysis: {}".format(e))
        completed = False
    return completed

if __name__ == '__main__':
    with initialize(config_path="../conf", job_name="step3_analysis_nn_classification_error"):
        cfg = compose(config_name="config", overrides=["data.project_dir=/home/namj/ShapeY", 'network=mixer_b16_224', 'data=ShapeY200CR'])
        get_nn_classification_error(cfg)