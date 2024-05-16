import os
import sys
#from mega_nerf.get_meganerf import load_mega_nerf
import random
from torch import optim
import torchvision
from pose_regressor.dataset_loaders.load_mega_nerf import load_mega_nerf_dataloader
from pose_regressor.script.utils import freeze_bn_layer, freeze_bn_layer_train
from tqdm import tqdm
from pose_regressor.script.callbacks import EarlyStopping
from pose_regressor.script.dfnet import DFNet, DFNet_s
from pose_regressor.script.misc import *
from pose_regressor.script.options import config_parser
from torch.utils.tensorboard import SummaryWriter
from nerfacto_loader import load_model
from nerfacto_loader import get_params
from nerfacto_loader import vizualize
from PIL import Image
from torchvision.transforms import ToTensor
parser = config_parser()
args = parser.parse_args()
basedir = args.basedir
expname = 'output'

factor = 2
transform_path = '/home/asterisreppas/hal_nerf/workspace/colmap_output/colmap'
checkpoint_path = '/home/asterisreppas/hal_nerf/workspace/weight.ckpt'
nerfacto_model = load_model(transform_path, checkpoint_path, factor) 
nerfacto_params = get_params(transform_path, factor)


poses_dir = '/home/asterisreppas/pose_regressor_locnerf/logs/iw_old_1/output_poses'

poses = []
for pose_filename in os.listdir(poses_dir):
    if pose_filename.endswith('.pt'):
        pose_path = os.path.join(poses_dir, pose_filename)
        pose = torch.load(pose_path)
        pose = torch.tensor(pose)
        poses.append(pose)

poses = torch.stack(poses) if poses else None
print(poses[1].squeeze().shape)

predict_view = []
for i in(range(len(poses))):
    rendered_output_image = vizualize(poses[i].squeeze(), nerfacto_model, nerfacto_params)
    # Convert the numpy array to a PyTorch tensor
    rendered_output_tensor = torch.from_numpy(rendered_output_image).type(torch.float32)
    predict_view.append(rendered_output_tensor)

                
predict_view = torch.stack(predict_view).detach()

unloader = torchvision.transforms.ToPILImage()
os.makedirs(os.path.join(basedir, expname, 'output_images_predicted'), exist_ok=True)

writer =  SummaryWriter()
for i in range(predict_view.shape[0]):
    vis = predict_view[i].permute(2, 0, 1)
    writer.add_image("virtual_images", vis, i)
    image = vis
    image = unloader(image)
    image_path = os.path.join(basedir, expname, 'output_images_predicted', f'{i}.jpg')
    image.save(image_path)