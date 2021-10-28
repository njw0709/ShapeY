import torchvision.datasets as datasets
from torch.utils.data import Dataset
from itertools import combinations
import math

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
        tuple_with_path = (original_tuple + (path,))
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


class OriginalandPostProcessedPairsDataset(Dataset):
    def __init__(self, original_feat_dataset, postprocessed_feat_dataset):
        self.original = original_feat_dataset
        self.postprocessed = postprocessed_feat_dataset

    def __getitem__(self, index):
        idx1 = int(math.floor(index / len(self.postprocessed)))
        idx2 = index % len(self.postprocessed)
        s1 = self.original[idx1]
        s2 = self.postprocessed[idx2]
        return (idx1, s1), (idx2, s2)

    def __len__(self):
        return len(self.original)**2



