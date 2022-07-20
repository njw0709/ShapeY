import pytest
import h5py
import os
import shutil
import numpy as np
from shapey.dataprocess.raw_data import ImgCorrelationDataProcessorV2
import logging

FIXTURE_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(FIXTURE_DIR, 'test_data')


# @pytest.fixture(scope="session")
# def h5_feats_data():
#     tmp_file_path = os.path.join(DATA_DIR, 'tmp', 'resnet50_feats.h5')
#     shutil.copy(os.path.join(DATA_DIR, 'resnet50_feats.h5'), tmp_file_path)
#     hdfstore = h5py.File(tmp_file_path)
#     yield hdfstore
#     hdfstore.close()
#     os.remove(tmp_file_path)

# @pytest.fixture(scope="session")
# def h5_corrs_data():
#     tmp_file_path = os.path.join(DATA_DIR, 'tmp', 'resnet50_base.h5')
#     shutil.copy(os.path.join(DATA_DIR, 'resnet50_base.h5'), tmp_file_path)
#     hdfstore = h5py.File(tmp_file_path)
#     yield hdfstore
#     hdfstore.close()
#     os.remove(tmp_file_path)

@pytest.fixture(scope="session")
def fake_imgnames():
    # create fake image names and their vectors
    objnames = ['coffee', 'icecream', 'jam', 'hairdryer', 'microwave']
    axes = ImgCorrelationDataProcessorV2.generate_axes_of_interest()
    series = ['{}{}'.format(ax, '{:02d}'.format(num)) for num in range(0,11) for ax in axes]
    original_stored_imgname = ['{}-{}.png'.format(obj, ser) for obj in objnames for ser in series]
    return original_stored_imgname

@pytest.fixture(scope="session")
def h5_feats_small_fake(fake_imgnames):
    # create a fake dataset
    tmp_file_path = os.path.join(DATA_DIR, 'tmp', 'small_fake.h5')
    hdfstore = h5py.File(tmp_file_path, 'w')
    feature_group_key = '/feature_output'
    hdfstore.create_group(feature_group_key)
    feature_output_key = feature_group_key + '/output'
    imgname_key = feature_group_key + '/imgname'

    original_stored_imgname = fake_imgnames
    original_stored_feat = np.random.rand(len(original_stored_imgname), 100)

    hdfstore.create_dataset(imgname_key, data=np.array(original_stored_imgname).astype('S'))
    hdfstore.create_dataset(feature_output_key, data=original_stored_feat)
    yield hdfstore
    hdfstore.close()
    os.remove(tmp_file_path)



    

