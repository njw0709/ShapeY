from shapey.dataprocess.raw_data import ImgCorrelationDataProcessorV2
import shutil
import argparse
import os
import h5py

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='passes data directory and output file path')
    parser.add_argument('--input_dir', type=str, default=os.path.join(DATA_DIR, 'intermediate', 'your_feature.h5'))
    parser.add_argument('--output_dir', type=str, default=os.path.join(DATA_DIR, 'processed', 'your_feature_objnum_exp.h5'))


    args = parser.parse_args()
    print(args)

    input_name = args.input_dir
    output_name = args.output_dir

    # first check and copy the h5 file
    if not os.path.exists(output_name):
        shutil.copyfile(input_name, output_name)
    
    with h5py.File(output_name, 'a') as hdfstore:
        data_processor = ImgCorrelationDataProcessorV2(hdfstore)
        for num_obj in [20, 40, 80, 160, 200]:
            data_processor.cut_corrmat(hdfstore, num_obj)
            data_processor.exclusion_distance_analysis(hdfstore, num_objs=num_obj)