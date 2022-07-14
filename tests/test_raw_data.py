import pytest
from shapey.dataprocess.raw_data import compute_distance_and_save
from shapey.utils.customdataset import HDFDataset, PermutationPairsDataset


@pytest.fixture(scope="module")
def hdf_dataset(h5_feats_small_fake):
    #pull features
    feature_group_key = '/feature_output'
    feature_output_key = feature_group_key + '/output'
    original_features = h5_feats_small_fake[feature_output_key]
    hdfdataset = HDFDataset(original_features)
    return hdfdataset


@pytest.fixture(scope="module")
def permutation_dataset(hdf_dataset):
    perm_dataset = PermutationPairsDataset(hdf_dataset)
    return perm_dataset


def test_complete_run_compute_distance_and_save(
    h5_feats_small_fake, permutation_dataset
):
    # load features
    feature_group_key = "/feature_output"
    feature_output_key = feature_group_key + "/output"
    original_features = h5_feats_small_fake[feature_output_key]

    # create correlation matrix (placeholder)
    num_features = len(original_features)
    corrval_key = "/pairwise_correlation"
    corrval_key_original = corrval_key + "/original"
    h5_feats_small_fake.create_dataset(
        corrval_key_original, shape=(num_features, num_features)
    )

    completed, idx1, idx2 = compute_distance_and_save(
        permutation_dataset,
        h5_feats_small_fake,
        corrval_key_original,
        batch_size=10000,
        num_workers=5
    )

    assert completed == True
    assert idx1.max() == num_features - 1
    assert idx2.max() == num_features - 1


<<<<<<< HEAD
#TODO: implement testing for handling repeating batch error
def test_repeating_correlation_batch():
    assert 1
=======
>>>>>>> ec691c8ffc73f95ac104a9b22aea431869f6a569
