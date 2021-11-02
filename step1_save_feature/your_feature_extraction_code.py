from typing import Tuple
from resnet_wider import resnet50x1, resnet50x2, resnet50x4
import torch
from shapey.utils.modelutils import GetModelIntermediateLayer
import torchvision.transforms as transforms
import torch
from shapey.utils.customdataset import ImageFolderWithPaths
import tqdm

def your_feature_output_code(datadir: str) -> Tuple[list, list]:
    ## Takes in the dataset directory as string, and outputs a list with all image names, and a list with all extracted features.

    raise NotImplementedError

## example feature extraction code for resnet based simclr models
def extract_features_torch(datadir: str, model) -> Tuple[list, list]:
    ## Takes in the dataset directory as string, and outputs a list with all image names, and a list with all extracted features.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    dataset = ImageFolderWithPaths(datadir, transforms.Compose([
                                            transforms.Resize(224),
                                            transforms.ToTensor(),
                                            normalize
                                        ]))

    data_loader = torch.utils.data.DataLoader(dataset,
                    batch_size=1, shuffle=False,
                    num_workers=4, pin_memory=True)

    # compute features
    original_stored_imgname = []
    original_stored_feat = []
    for s in tqdm(data_loader):
        img1, _, fname1 = s
        fname1 = fname1[0].split('/')[-1]
        output1 = model(img1.cuda())
        output1 = torch.flatten(output1)
        output1_store = output1.cpu().data.numpy()
        original_stored_imgname.append(fname1)
        original_stored_feat.append(output1_store)
    return original_stored_imgname, original_stored_feat

def simclr_resnet(width: int):
    if width == 1:
        model = resnet50x1()
        model.load_state_dict(torch.load('../simclr/resnet50x1.pth'))
    elif width == 2:
        model = resnet50x2()
        model.load_state_dict(torch.load('../simclr/resnet50x2.pth'))
    elif width == 4:
        model = resnet50x4()
        model.load_state_dict(torch.load('../simclr/resnet50x4.pth'))
    else:
        raise ValueError("Invalid width")
    model_rep = GetModelIntermediateLayer(model, -1)
    model_rep.cuda().eval()
    return model_rep
