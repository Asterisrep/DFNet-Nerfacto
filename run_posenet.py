
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


# Run your code again to see the detailed error log


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)
parser = config_parser()
args = parser.parse_args()


def train_on_batch(args, rgbs, poses, feat_model, dset_size, FeatureLoss, optimizer, hwf):
    ''' core training loop for featurenet'''
    feat_model.train()
    H, W, focal = hwf
    H, W = int(H), int(W)
    if args.freezeBN:
        feat_model = freeze_bn_layer_train(feat_model)

    train_loss_epoch = []
    select_inds = np.random.choice(dset_size, size=[dset_size], replace=False)  # (N_rand,)

    batch_size=args.featurenet_batch_size # manual setting, use smaller batch size like featurenet_batch_size = 4 if OOM
    if dset_size % batch_size == 0:
        N_iters = dset_size//batch_size
    else:
        N_iters = dset_size//batch_size + 1
    i_batch = 0

    for i in range(0, N_iters):
        if i_batch + batch_size > dset_size:
            i_batch = 0
            break
        i_inds = select_inds[i_batch:i_batch+batch_size]
        i_batch = i_batch + batch_size

        # convert input shape to [B, 3, H, W]
        # target_in = targets[i_inds].clone().permute(0,3,1,2).to(device)
        rgb_in = rgbs[i_inds].clone().permute(0,3,1,2).to(device)
        pose = poses[i_inds].clone().reshape(batch_size, 12).to(device)
        # pose = torch.cat([pose, pose]) # double gt pose tensor

        features, predict_pose = feat_model(rgb_in, False, upsampleH=H, upsampleW=W) # features: (1, [2, B, C, H, W])
        #pose[:, [3, 7, 11]] *= args.map_scale
        loss = PoseLoss(args, predict_pose, pose, device)  # target

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_epoch.append(loss.item())
    train_loss = np.mean(train_loss_epoch)
    return train_loss


def train_on_batch_with_random_view_synthesis(args, targets, rgbs, poses, virtue_view, poses_perturb, feat_model, dset_size, FeatureLoss, optimizer, hwf):
    ''' we implement random view synthesis for generating more views to help training posenet '''
    feat_model.train()

    H, W, focal = hwf
    H, W = int(H), int(W)

    if args.freezeBN:
        feat_model = freeze_bn_layer_train(feat_model)

    train_loss_epoch = []

    # random generate batch_size of idx
    select_inds = np.random.choice(dset_size, size=[dset_size], replace=False)  # (N_rand,)

    batch_size = args.featurenet_batch_size
    # manual setting, use smaller batch size like featurenet_batch_size = 4 if OOM
    if dset_size % batch_size == 0:
        N_iters = dset_size//batch_size
    else:
        N_iters = dset_size//batch_size + 1
    
    i_batch = 0
    for i in range(0, N_iters):
        if i_batch + batch_size > dset_size:
            break
        i_inds = select_inds[i_batch:i_batch+batch_size]
        i_batch = i_batch + batch_size

        # convert input shape to [B, 3, H, W]
        target_in = targets[i_inds].clone().permute(0,3,1,2).to(device)
        rgb_in = rgbs[i_inds].clone().permute(0,3,1,2).to(device)
        pose = poses[i_inds].clone().reshape(batch_size, 12).to(device)
        rgb_perturb = virtue_view[i_inds].clone().permute(0,3,1,2).to(device)
        pose_perturb = poses_perturb[i_inds].clone().reshape(batch_size, 12).to(device)

        # inference feature model for GT and nerf image
        #print(target_in[0].shape, rgb_in[0].shape)
        pose = torch.cat([pose, pose]) # double gt pose tensor
        features, predict_pose = feat_model(torch.cat([target_in, rgb_in]), return_feature=True, upsampleH=H, upsampleW=W) # features: (1, [2, B, C, H, W])

        # get features_target and features_rgb
        if args.DFNet:
            features_target = features[0] # [3, B, C, H, W]
            features_rgb = features[1]

        loss_pose = PoseLoss(args, predict_pose, pose, device) # target

        if args.tripletloss:
            loss_f = triplet_loss_hard_negative_mining_plus(features_rgb, features_target, margin=args.triplet_margin)
        else:
            loss_f = FeatureLoss(features_rgb, features_target) # feature Maybe change to s2d-ce loss

        # inference model for RVS image
        _, virtue_pose = feat_model(rgb_perturb.to(device), False)

        # add relative pose loss here. TODO: This FeatureLoss is nn.MSE. Should be fixed later
        loss_pose_perturb = PoseLoss(args, virtue_pose, pose_perturb, device)
        loss = args.combine_loss_w[0]*loss_pose + args.combine_loss_w[1]*loss_f + args.combine_loss_w[2]*loss_pose_perturb

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss_epoch.append(loss.item())
    train_loss = np.mean(train_loss_epoch)
    return train_loss


def train_feature(args, train_dl, val_dl, test_dl, hwf, i_split, nerfacto_model, nerfacto_params):
    writer = SummaryWriter()
    # # load pretrained PoseNet model
    if args.DFNet_s:
        feat_model = DFNet_s()
    else:
        feat_model = DFNet()
    
    if args.pretrain_model_path != '':
        print("load posenet from ", args.pretrain_model_path)
        feat_model.load_state_dict(torch.load(args.pretrain_model_path))


    
    # # Freeze BN to not updating gamma and beta
    if args.freezeBN:
        feat_model = freeze_bn_layer(feat_model)

    feat_model.to(device)
    # summary(feat_model, (3, 240, 427))

    # set optimizer
    optimizer = optim.Adam(feat_model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.95, patience=args.patience[1], verbose=True)

    # set callbacks parameters
    early_stopping = EarlyStopping(args, patience=args.patience[0], verbose=False, delta=0.000001)

    # loss function
    loss_func = nn.MSELoss(reduction='mean')
    i_train, i_val, i_test = i_split
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    N_epoch = args.epochs + 1  # epoch
    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)


    # if args.eval:
    #     sample_size = 52
    #     feat_model.eval()
    #     #get_error_in_q(args, test_dl, feat_model, len(val_dl.dataset), device, batch_size=1)
    #     get_render_error_in_q(args, feat_model, sample_size, device, targets, rgbs, poses, batch_size=1)
    #     sys.exit()

    # import torch

    
    #### AR START ####
    #custom evaluation synthetic dataset
    if args.eval:
        sample_size = 81
        feat_model.eval()
        rgb_dir = '/home/asterisreppas/pose_regressor_locnerf/logs/Synthetic_Testsets/iw_old_testset_1_180/output_img_virtual_test'
        poses_dir = '/home/asterisreppas/pose_regressor_locnerf/logs/Synthetic_Testsets/iw_old_testset_1_180/output_poses_virtual_test'
        
        # Load rgbs (Assuming they are images in a directory)
        rgbs = []
        for img_filename in sorted(os.listdir(rgb_dir)):
            if img_filename.endswith('.jpg'):
                img_path = os.path.join(rgb_dir, img_filename)
                image = Image.open(img_path).convert('RGB')
                #image = transform(image)  # Apply any required transformations
                rgbs.append(image)
        
        # Load poses (Assuming they are saved as '.pt' files in a directory)
        poses = []
        for pose_filename in sorted(os.listdir(poses_dir)):
            if pose_filename.endswith('.pt'):
                pose_path = os.path.join(poses_dir, pose_filename)
                pose = torch.load(pose_path)
                poses.append(pose)
        
        # Convert lists to tensors if necessary
        tensor_rgbs = [ToTensor()(image) for image in rgbs]
        rgbs = torch.stack(tensor_rgbs).detach() if tensor_rgbs else None
        poses = torch.stack(poses) if poses else None
        # print('edo koita tora')
        # print(rgbs.shape, poses.shape)
        # print(rgbs, poses)
        
        batch_size = 1
        # Now you can pass the rgbs and poses to the function
        get_render_error_in_q(args, feat_model, sample_size, device, rgbs, poses, batch_size, basedir, expname)
        
        sys.exit()

        #### AR END ####



        
    rgb_list = []
    pose_list = []
    for data, pose, img_idx in train_dl:
        # convert to H W C
        rgb_list.append(data[0].permute(1, 2, 0))
        pose_list.append(pose[0].reshape(3, 4))
    rgbs = torch.stack(rgb_list).detach()
    poses = torch.stack(pose_list).detach()
    # rgbs = targets.clone()
    #### AR start ####
    targets, rgbs, poses, img_idxs = render_nerfw_imgs(args, train_dl, hwf, device, nerfacto_model, nerfacto_params)
    #### AR test ####
   
    
    #visualize rgb
    unloader = torchvision.transforms.ToPILImage()
    os.makedirs(os.path.join(basedir, expname, 'output_img_rgb'), exist_ok=True)
    for i in range(rgbs.shape[0]):
        vis = rgbs[i].permute(2, 0, 1)
        writer.add_image("rgb_images", vis, i)
        image = vis.clone()  # clone the tensor
        #image = image.squeeze(0)  # remove the fake batch dimension
        image = unloader(image)
        image.save(os.path.join(basedir, expname, 'output_img_rgb/') + str(i) + '.jpg')


    dset_size = len(train_dl.dataset)
    # clean GPU memory before testing, try to avoid OOM
    torch.cuda.empty_cache()

    model_log = tqdm(total=0, position=1, bar_format='{desc}')
    for epoch in tqdm(range(N_epoch), desc='epochs'):

        if args.random_view_synthesis:
            isRVS = epoch % args.rvs_refresh_rate == 0  # decide if to resynthesis new views

            if isRVS:
                # random sample virtual camera locations, todo:
                rand_trans = args.rvs_trans
                rand_rot = args.rvs_rotation

                # determine bounding box
                b_min = [poses[:, 0, 3].min()-args.d_max, poses[:, 1, 3].min()-args.d_max, poses[:, 2, 3].min()-args.d_max]
                b_max = [poses[:, 0, 3].max()+args.d_max, poses[:, 1, 3].max()+args.d_max, poses[:, 2, 3].max()+args.d_max]

                # 扰动
                poses_perturb = poses.clone().numpy()
                for i in range(dset_size):
                    poses_perturb[i] = perturb_single_render_pose(poses_perturb[i], rand_trans, rand_rot)
                    for j in range(3):
                        if poses_perturb[i,j,3] < b_min[j]:
                            poses_perturb[i,j,3] = b_min[j]
                        elif poses_perturb[i,j,3]> b_max[j]:
                            poses_perturb[i,j,3] = b_max[j]

                
                

                poses_perturb = torch.Tensor(poses_perturb)  # [B, 3, 4]
                tqdm.write("renders RVS...")
                #virtue_view = mega_nerf_model.render_virtual_meganerf_imgs(args, poses_perturb, hwf, device)

                # Render the perturbed poses to augment train dataset

                ### AR START ###
                virtue_view = []
                for i in(range(len(poses_perturb))):
                    rendered_image = vizualize(poses_perturb[i], nerfacto_model, nerfacto_params)
    # Convert the numpy array to a PyTorch tensor
                    #rendered_tensor = torch.from_numpy(rendered_image).type(torch.float32)
                    virtue_view.append(rendered_image)

                
                virtue_view = torch.stack(virtue_view).detach()


                


                ##### AR END #####
                '''
                visualization
                
                '''

                unloader = torchvision.transforms.ToPILImage()
                os.makedirs(os.path.join(basedir, expname, 'output_img_virtual'), exist_ok=True)
                os.makedirs(os.path.join(basedir, expname, 'output_poses_virtual'), exist_ok=True)  # Create directory for poses

                for i in range(virtue_view.shape[0]):
                    vis = virtue_view[i].permute(2, 0, 1)
                    writer.add_image("virtual_images", vis, i)
                    image = vis
                    image = unloader(image)
                    image_path = os.path.join(basedir, expname, 'output_img_virtual', f'{i}.jpg')
                    image.save(image_path)
                    
                    # Save the corresponding pose for this image
                    pose_path = os.path.join(basedir, expname, 'output_poses_virtual', f'pose_{i}.pt')
                    torch.save(poses_perturb[i], pose_path)  # Assuming poses_perturb[i] is the correct indexing

            

                

            train_loss = train_on_batch_with_random_view_synthesis(args, targets, rgbs, poses, virtue_view, poses_perturb, feat_model, dset_size, loss_func, optimizer, hwf)
            
        else:
            train_loss = train_on_batch(args, rgbs, poses, feat_model, dset_size, loss_func, optimizer, hwf)

        feat_model.eval()
        val_loss_epoch = []
        for data, pose, _ in val_dl:
            inputs = data.to(device)
            labels = pose.to(device)
            #print(inputs, inputs.shape)
            # labels = labels.view(1, 12)
            # pose loss
            #labels[:, [3, 7, 11]] *= args.map_scale
            _, predict = feat_model(inputs)
            loss = loss_func(predict, labels)
            val_loss_epoch.append(loss.item())
        val_loss = np.mean(val_loss_epoch)

        # reduce LR on plateau
        scheduler.step(val_loss)

        # logging
        tqdm.write('At epoch {0:6d} : train loss: {1:.6f}, val loss: {2:.6f}'.format(epoch, train_loss, val_loss))
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/test', val_loss, epoch)

        # check wether to early stop
        early_stopping(val_loss, feat_model, epoch=epoch, save_multiple=(not args.no_save_multiple), save_all=args.save_all_ckpt)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        model_log.set_description_str(f'Best val loss: {early_stopping.val_loss_min:.4f}')
        if epoch % args.i_eval == 0:
            mediaT, mediaR, meanT, meanR = get_error_in_q(args, test_dl, feat_model, len(test_dl.dataset), device, batch_size=1)
            writer.add_scalar("Test/mediaTranslation", mediaT, epoch)
            writer.add_scalar("Test/mediaRotation", mediaR, epoch)
            writer.add_scalar("Test/meanTranslation", meanT, epoch)
            writer.add_scalar("Test/meanRotation", meanR, epoch)

    writer.close()    # global_step += 1
    return


def train():
    #print(parser.format_values()) ####  AR START ####
    #mega_nerf_model = load_mega_nerf(args.exp_name, args.datadir, args.config_file, args.container_path)
    factor = 2
    transform_path = '/root/dfnet_nerfacto/workspace/colmap_output/colmap'
    checkpoint_path = '/root/dfnet_nerfacto/workspace/weight.ckpt'
    nerfacto_model = load_model(transform_path, checkpoint_path, factor) 
    nerfacto_params = get_params(transform_path, factor)
    #### AR END ####
    #assert args.dataset_type == 'mega_nerf'
    train_dl, val_dl, test_dl, hwf, i_split = load_mega_nerf_dataloader(args, nerfacto_params)
    train_feature(args, train_dl, val_dl, test_dl, hwf, i_split, nerfacto_model, nerfacto_params)
    return


if __name__ == "__main__":
    train()
