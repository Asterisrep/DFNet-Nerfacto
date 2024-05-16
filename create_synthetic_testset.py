import os
import sys
#from mega_nerf.get_meganerf import load_mega_nerf
import random
from torch import optim
import torchvision
from pose_regressor.dataset_loaders.load_mega_nerf import load_mega_nerf_dataloader
from pose_regressor.script.utils import freeze_bn_layer, freeze_bn_layer_train
from tqdm import tqdm
#from callbacks import EarlyStopping
from pose_regressor.script.dfnet import DFNet, DFNet_s
from pose_regressor.script.misc import *
from pose_regressor.script.options import config_parser
from torch.utils.tensorboard import SummaryWriter
from nerfacto_loader import load_model
from nerfacto_loader import get_params

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
parser = config_parser()
args = parser.parse_args()


# Load nerfacto pretrained model and get input dataset
factor = 2
transform_path = '/home/asterisreppas/hal_nerf/workspace/colmap_output/colmap'
checkpoint_path = '/home/asterisreppas/hal_nerf/workspace/weight.ckpt'
basedir = args.basedir
expname = 'iw_old_testset_1_180'
nerfacto_model = load_model(transform_path, checkpoint_path, factor) 
nerfacto_params = get_params(transform_path, factor)
train_dl, val_dl, test_dl, hwf, i_split = load_mega_nerf_dataloader(args, nerfacto_params)
rgb_list = []
pose_list = []
for data, pose, img_idx in train_dl:
        # convert to H W C
    rgb_list.append(data[0].permute(1, 2, 0))
    pose_list.append(pose[0].reshape(3, 4))
rgbs = torch.stack(rgb_list).detach()
poses = torch.stack(pose_list).detach()


# Give the translation and rotation parametres with which ground truth poses will be perturbed
rand_trans = 1
rand_rot = 180
# determine bounding box
b_min = [poses[:, 0, 3].min()-args.d_max, poses[:, 1, 3].min()-args.d_max, poses[:, 2, 3].min()-args.d_max]
b_max = [poses[:, 0, 3].max()+args.d_max, poses[:, 1, 3].max()+args.d_max, poses[:, 2, 3].max()+args.d_max]



# Create new - synthetic poses
dset_size = len(train_dl.dataset)
k = 0
num_perturbed_poses = math.ceil(dset_size / 10)
poses_perturb_test = np.zeros((num_perturbed_poses, 3, 4))

poses_perturb_initial = poses.clone().numpy()
for i in range(dset_size):
    if i % 10 == 0:  # Check if the current index is a multiple of 40
                        # Only perturb every 40th pose
        poses_perturb_test[k] = perturb_single_render_pose(poses_perturb_initial[i], rand_trans, rand_rot)
                
                    # Perform bounds checking for every pose, regardless of perturbation
        for j in range(3):  # For x, y, z translation components
            if poses_perturb_test[k, j, 3] < b_min[j]:
                poses_perturb_test[k, j, 3] = b_min[j]
            elif poses_perturb_test[k, j, 3] > b_max[j]:
                poses_perturb_test[k, j, 3] = b_max[j]

        #print(poses_perturb_initial[i], poses_perturb_test[k])            
        k += 1
       



# Render virtual poses
poses_perturb_test = torch.Tensor(poses_perturb_test)

print(poses_perturb_test[1],poses_perturb_test[1].shape)
virtue_view_test = []
for i in(range(len(poses_perturb_test))):
    rendered_image_test = vizualize(poses_perturb_test[i], nerfacto_model, nerfacto_params)
    # Convert the numpy array to a PyTorch tensor
    rendered_tensor_test = torch.from_numpy(rendered_image_test).type(torch.float32)
    virtue_view_test.append(rendered_tensor_test)

            
            
virtue_view_test = torch.stack(virtue_view_test).detach()
print(virtue_view_test)



#Save them
unloader = torchvision.transforms.ToPILImage()
os.makedirs(os.path.join(basedir, expname, 'output_img_virtual_test'), exist_ok=True)
os.makedirs(os.path.join(basedir, expname, 'output_poses_virtual_test'), exist_ok=True)  # Create directory for poses

writer =  SummaryWriter()
for i in range(virtue_view_test.shape[0]):
    vis = virtue_view_test[i].permute(2, 0, 1)
    writer.add_image("virtual_images_test", vis, i)
    image = vis
    image = unloader(image)
    image_path = os.path.join(basedir, expname, 'output_img_virtual_test', f'{i}.jpg')
    image.save(image_path)
                    
                    # Save the corresponding pose for this image
    pose_path = os.path.join(basedir, expname, 'output_poses_virtual_test', f'pose_{i}.pt')
    torch.save(poses_perturb_test[i], pose_path)  # Assuming poses_perturb[i] is the correct indexing