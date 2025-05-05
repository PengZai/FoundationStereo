# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
from Utils import *
from core.foundation_stereo import *
import matplotlib
import cv2

if __name__=="__main__":
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--left_file', default="/root/catkin_ws/src/modules_vins/examples/FoundationStereo/apple-left.png", type=str)
  parser.add_argument('--right_file', default="/root/catkin_ws/src/modules_vins/examples/FoundationStereo/apple-right.png", type=str)
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K_apple_zed.txt', type=str, help='camera intrinsic matrix and baseline file')
  # parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/23-51-11/model_best_bp2.pth', type=str, help='pretrained model path')
  parser.add_argument('--ckpt_dir', default=f'{code_dir}/../pretrained_models/11-33-40/model_best_bp2.pth', type=str, help='pretrained model path')
  parser.add_argument('--out_dir', default="/root/catkin_ws/src/modules_vins/examples/FoundationStereo/apple-left-small", type=str, help='the directory to save results')
  parser.add_argument('--scale', default=1, type=float, help='downsize the image by scale, must be <=1')
  parser.add_argument('--hiera', default=0, type=int, help='hierarchical inference (only needed for high-resolution images (>1K))')
  parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
  parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
  parser.add_argument('--remove_invisible', default=1, type=int, help='remove non-overlapping observations between left and right images from point cloud, so the remaining points are more reliable')

  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)
  os.makedirs(args.out_dir, exist_ok=True)

  ckpt_dir = args.ckpt_dir
  cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
  if 'vit_size' not in cfg:
    cfg['vit_size'] = 'vitl'
  for k in args.__dict__:
    cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)
  logging.info(f"args:\n{args}")
  logging.info(f"Using pretrained model from {ckpt_dir}")

  model = FoundationStereo(args)

  ckpt = torch.load(ckpt_dir)
  logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
  model.load_state_dict(ckpt['model'])

  model.cuda()
  model.eval()

  code_dir = os.path.dirname(os.path.realpath(__file__))
  img0 = imageio.imread(args.left_file)
  img1 = imageio.imread(args.right_file)
  scale = args.scale
  assert scale<=1, "scale must be <=1"
  img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
  img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
  H,W = img0.shape[:2]
  img0_ori = img0.copy()
  logging.info(f"img0: {img0.shape}")

  img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
  img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
  padder = InputPadder(img0.shape, divis_by=32, force_square=False)
  img0, img1 = padder.pad(img0, img1)
  
  with torch.cuda.amp.autocast(True):
    if not args.hiera:
      disp = model(img0, img1)
      traced_script_module = torch.jit.trace(model, [img0, img1])
      traced_script_module.save(ckpt_dir+".traced.pt")




  