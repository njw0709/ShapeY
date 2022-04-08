from typing import Tuple
from resnet_wider import resnet50x1, resnet50x2, resnet50x4
import torch
from shapey.utils.modelutils import GetModelIntermediateLayer
import torchvision.transforms as transforms
import torch
from pytorch_pretrained_vit import ViT
from shapey.utils.customdataset import ImageFolderWithPaths
from tqdm import tqdm
import timm
import os

PROJECT_DIR = os.path.join(os.path.dirname(__file__), '..')


def your_feature_output_code(datadir: str) -> Tuple[list, list]:
    ## Takes in the dataset directory as string, and outputs a list with all image names, and a list with all extracted features.

    raise NotImplementedError

## example feature extraction code for resnet based simclr models
def extract_features_torch(datadir: str, model, input_img_size:int = 244) -> Tuple[list, list]:
    ## Takes in the dataset directory as string, and outputs a list with all image names, and a list with all extracted features.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    dataset = ImageFolderWithPaths(datadir, transforms.Compose([
                                            transforms.Resize(input_img_size),
                                            transforms.ToTensor(),
                                            normalize
                                        ]))

    data_loader = torch.utils.data.DataLoader(dataset,
                    batch_size=1, shuffle=False,
                    num_workers=0, pin_memory=True)

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

def extract_features_torch_with_hooks(datadir: str, model, final_layer_name: str, input_img_size: int = 244) -> Tuple[list, list]:
    ## Takes in the dataset directory as string, and outputs a list with all image names, and a list with all extracted features.
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    dataset = ImageFolderWithPaths(datadir, transforms.Compose([
                                            transforms.Resize(input_img_size),
                                            transforms.ToTensor(),
                                            normalize
                                        ]))

    data_loader = torch.utils.data.DataLoader(dataset,
                    batch_size=1, shuffle=False,
                    num_workers=0, pin_memory=True)

    features = {}
    def get_features(name):
        def hook(model, input, output):
            if type(input) == torch.Tensor:
                features[name] = input.detach()
            elif type(input) == tuple:
                if len(input) == 1:
                    features[name] = input[0].detach()
                else:
                    raise NotImplementedError('input is a tuple with size larger than one')
            else:
                raise NotImplementedError('input is not a tensor or a tuple of tensors')
        return hook
    
    exec("model." + final_layer_name + ".register_forward_hook(get_features('" + final_layer_name + "'))")
    
    # compute features
    original_stored_imgname = []
    original_stored_feat = []
    for s in tqdm(data_loader):
        img1, _, fname1 = s
        fname1 = fname1[0].split('/')[-1]
        _ = model(img1.cuda())
        prelogit = features[final_layer_name]
        output = torch.flatten(prelogit)
        output = output.cpu().data.numpy()
        original_stored_imgname.append(fname1)
        original_stored_feat.append(output)
    return original_stored_imgname, original_stored_feat


def simclr_resnet(width: int):
    if width == 1:
        model = resnet50x1()
        sd = torch.load(os.path.join(PROJECT_DIR, 'simclr', 'resnet-1x.pth'), map_location='cpu')
        model.load_state_dict(sd['state_dict'])
    elif width == 2:
        model = resnet50x2()
        sd = torch.load(os.path.join(PROJECT_DIR, 'simclr', 'resnet-2x.pth'), map_location='cpu')
        model.load_state_dict(sd['state_dict'])
    elif width == 4:
        model = resnet50x4()
        sd = torch.load(os.path.join(PROJECT_DIR, 'simclr', 'resnet-4x.pth'), map_location='cpu')
        model.load_state_dict(sd['state_dict'])
    else:
        raise ValueError("Invalid width")
    return model

def timm_get_model(model_name: str):
    simclr_models = ['simclr_v1_resnet50x1', 'simclr_v1_resnet50x2', 'simclr_v1_resnet50x4']
    if model_name in simclr_models:
        model = simclr_resnet(int(model_name.split('x')[-1]))
    else:
        model = timm.create_model(model_name, pretrained=True)
    model.cuda().eval()
    return model
