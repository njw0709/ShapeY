from typing import List
import pytest
import numpy as np
from shapey.utils.customdataset import PermutationDatasetWithNNBatches, HDFDataset
from test_raw_data import hdf_dataset


"""
 Test LSH nearest neighbor batches 
 -> pytorch dataset that computes correlation for those batches and saves out as sparse matrix
"""

@pytest.fixture(scope="module")
def fake_nn_batch_list_small(fake_imgnames: list):
    num_images = len(fake_imgnames)
    num_elements_per_batch = 300
    assert num_images > num_elements_per_batch
    #make sample lists
    num_elements_list = [num_elements_per_batch]*int(num_images/num_elements_per_batch)
    num_elements_list.append(num_images%num_elements_per_batch)

    idxs = np.arange(num_images)
    np.random.shuffle(idxs)
    batch_idxs = []
    for num_elem in num_elements_list:
        batch_idxs.append(idxs[:num_elem])
        idxs = idxs[num_elem:]
    return batch_idxs

# Check if the output of the custom dataset is correct
def test_output_permutation_dataset_with_nn_idx(fake_nn_batch_list_small, hdf_dataset):
    #rename variable
    batch_idxs = fake_nn_batch_list_small
    hdf_dataset_torch = hdf_dataset

    permutation_dataset_torch = PermutationDatasetWithNNBatches(hdf_dataset_torch, batch_idxs)
    
    #Check length
    batch_dims = [b.size for b in batch_idxs]
    batch_idx_cutoff = np.cumsum(np.array([bsize**2 for bsize in batch_dims]))
    batch_idx_cutoff = np.insert(batch_idx_cutoff, 0, 0)
    assert len(permutation_dataset_torch) == batch_idx_cutoff[-1]

    #Select random (permutation) index, and pull out the sample. Then check if it matches the expected sample.
    index = np.random.randint(len(permutation_dataset_torch))
    (idx1, s1), (idx2, s2) = permutation_dataset_torch[index]

    # convert perm index to index of the actual vector.
    def get_batch_id(index, batch_idx_cutoff):
        for i, cutoff in enumerate(batch_idx_cutoff):
            if index < cutoff:
                batch_id = i - 1
                break
        return batch_id
    
    def get_feat_id(index, batch_idx_cutoff, batch_dims, batch_id):
        # get within_batch_idx
        within_batch_idx = index - batch_idx_cutoff[batch_id]

        # convert to bidx1 and bidx2
        bidx1 = int(within_batch_idx/batch_dims[batch_id])
        bidx2 = within_batch_idx % batch_dims[batch_id]

        # convert to feature index
        feat_idx1 = batch_idxs[batch_id][bidx1]
        feat_idx2 = batch_idxs[batch_id][bidx2]
        return feat_idx1, feat_idx2
    
    batch_id = get_batch_id(index, batch_idx_cutoff)
    feat_idx1, feat_idx2 = get_feat_id(index, batch_idx_cutoff, batch_dims, batch_id)

    assert idx1 == feat_idx1
    assert idx2 == feat_idx2
    assert (s1 == hdf_dataset[feat_idx1]).all()
    assert (s2 == hdf_dataset[feat_idx2]).all()

    #test edge case where index = batch_size[0]
    index = batch_idx_cutoff[-1] - 1
    (idx1, s1), (idx2, s2) = permutation_dataset_torch[index]

    batch_id = get_batch_id(index, batch_idx_cutoff)
    feat_idx1, feat_idx2 = get_feat_id(index, batch_idx_cutoff, batch_dims, batch_id)

    assert idx1 == feat_idx1
    assert idx2 == feat_idx2
    assert (s1 == hdf_dataset[feat_idx1]).all()
    assert (s2 == hdf_dataset[feat_idx2]).all()

    #edge case 2
    index = batch_idx_cutoff[1]
    (idx1, s1), (idx2, s2) = permutation_dataset_torch[index]

    batch_id = get_batch_id(index, batch_idx_cutoff)
    assert batch_id == 1
    feat_idx1, feat_idx2 = get_feat_id(index, batch_idx_cutoff, batch_dims, batch_id)

    assert idx1 == feat_idx1
    assert idx2 == feat_idx2
    assert (s1 == hdf_dataset[feat_idx1]).all()
    assert (s2 == hdf_dataset[feat_idx2]).all()

