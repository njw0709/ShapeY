from shapey.dataprocess.raw_data import AllImgPairCorrelationData, PostProcessedAllImgPairCorrelationData
import argparse
import os

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='passes data directory and output file path')
    parser.add_argument('--input_dir', type=str, default=os.path.join(DATA_DIR, 'processed', 'your_feature.h5'))
    parser.add_argument('--output_dir', type=str, default=os.path.join(DATA_DIR, 'processed', 'your_feature.h5'))

    args = parser.parse_args()

    print(args)

    input_name = args.input_dir
    output_name = args.output_dir
    print('computing tuning curves...')
    if not args.postprocessed_comparison:
        resnet_output_allimgpairs = AllImgPairCorrelationData(input_name)
        resnet_output_allimgpairs.compute_tuning_curves()
    else:
        resnet_output_allimgpairs = PostProcessedAllImgPairCorrelationData(input_name)
        resnet_output_allimgpairs.compute_tuning_curves()
    print('done!')
    resnet_output_allimgpairs.hdfstore.close()
