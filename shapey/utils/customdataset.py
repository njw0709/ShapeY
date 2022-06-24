import torchvision.datasets as datasets
from torch.utils.data import Dataset
from itertools import combinations
import math
import psutil


class CombinationDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.comb = list(combinations(dataset, 2))

    def __getitem__(self, index):
        img1, img2 = self.comb[index]
        return img1, img2

    def __len__(self):
        return len(self.comb)

    def cut_dataset(self, index):
        self.comb = self.comb[index:]


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)
        return tuple_with_path


class FeatureTensorDatasetWithImgName(Dataset):
    def __init__(self, feature_tensor, img_name_array):
        self.feature_tensor = feature_tensor
        self.imgnames = img_name_array

    def __getitem__(self, index):
        feat = self.feature_tensor[index, :]
        imgname = self.imgnames[index]
        return imgname, feat

    def __len__(self):
        return len(self.imgnames)


class PermutationIndexDataset(Dataset):
    def __init__(self, datalen):
        self.datalen = datalen

    def __getitem__(self, index):
        idx1 = int(math.floor(index / self.datalen))
        idx2 = index % self.datalen
        return idx1, idx2


class OriginalandPostProcessedPairsDataset(Dataset):
    def __init__(self, original_feat_dataset, postprocessed_feat_dataset):
        self.original = original_feat_dataset
        self.postprocessed = postprocessed_feat_dataset
        self.datalen = len(self.postprocessed)

    def __getitem__(self, index):
        idx1 = int(math.floor(index / self.datalen))
        idx2 = index % self.datalen
        s1 = self.original[idx1]
        s2 = self.postprocessed[idx2]
        return (idx1, s1), (idx2, s2)

    def __len__(self):
        return len(self.original) ** 2


class PermutationPairsDataset(Dataset):
    def __init__(self, original_feat_dataset, postprocessed=None):
        self.original = original_feat_dataset
        self.datalen = len(self.original)
        self.postprocessed = postprocessed

    def __getitem__(self, index):
        idx1 = int(math.floor(index / self.datalen))
        idx2 = index % self.datalen
        s1 = self.original[idx1]
        if self.postprocessed is not None:
            s2 = self.postprocessed[idx2]
        else:
            s2 = self.original[idx2]
        return (idx1, s1), (idx2, s2)

    def __len__(self):
        return len(self.original) ** 2


class HDFDataset(Dataset):
    def __init__(self, hdfstore, mem_usage=0.85):
        self.hdfstore = hdfstore
        self.datalen = len(self.hdfstore)
        self.pull_data_to_cache(mem_usage)
        if not self.all_in_cache:
            print("initializing placeholder cache list")
            self.cache_length = int(
                psutil.virtual_memory().available * 0.85 / self.hdfstore[0].nbytes
            )
            self.in_cache_idx = [None] * self.cache_length
            self.in_cache = [None] * self.cache_length
            self.cache_counter = 0

    def __getitem__(self, index):
        if not self.all_in_cache:
            if index in self.in_cache_idx:
                return self.in_cache[self.in_cache_idx.index(index)]
            else:
                self.in_cache_idx[self.cache_counter] = index
                data = self.hdfstore[index]
                self.in_cache[self.cache_counter] = data
                self.cache_counter += 1
                self.cache_counter %= self.cache_length
                return data
        return self.hdfstore[index]

    def __len__(self):
        return self.datalen

    def pull_data_to_cache(self, mem_usage):
        single_row = self.hdfstore[0]
        if (
            psutil.virtual_memory().available * mem_usage
            < single_row.nbytes * self.datalen
        ):
            print("Not enough memory to pull data to cache")
            self.all_in_cache = False
        else:
            print("Pulling data to cache")
            self.hdfstore = self.hdfstore[:]
            self.all_in_cache = True
            print("Done pulling data to cache")
