#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import datetime

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import torch.nn.functional as F

from PIL import Image
import numpy as np
from utils.camera_utils import Camera

from utils.general_utils import PILtoTorch 
import copy

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import threading
import time
#WDD 5-13 
#一个新的类，用来将多个数据进行缓冲
class Viewpoint_Buffer:
    def __init__(self,index,white_background=True):
        self.index=index
        self.viewpoint_stack    =   []     #保存缓存的viewpoint数据
        self.is_loaded         =   False                #判断是否完成了数据的读取
        self.white_background   =   white_background
        self.current_index      =   0

        #self.lock = threading.Lock()

    #新增加数据
    def append(self,viewpoint_cam):
        self.viewpoint_stack.append(viewpoint_cam)

    #判断是否是最终的一个元素
    def is_last(self):
        return self.current_index==len(self.viewpoint_stack)-1
    #获得数据
    def pop(self): 
        viewpoint=self.viewpoint_stack[self.current_index] 
        self.current_index=self.current_index+1
        if self.current_index>=len(self.viewpoint_stack):
            self.current_index=0

        return viewpoint
    
    #获得读取shuju
    def get_is_load(self):
        return   self.is_loaded 
    

    #读取数据
    def load_images(self): 
        # 创建线程对象，目标函数是 read_data
        thread = threading.Thread(target=self.load_images_in_thr) 
        # 启动线程
        thread.start()

    #在多线程中读取数据
    def load_images_in_thr(self): 
        
        # 假设这里是加载图片的代码
        #with self.lock:  # 使用锁来确保线程安全

        self.is_loaded = False  
        for i in range(len(self.viewpoint_stack)):
            self.viewpoint_stack[i]=self.load_image_to_viewpoint(self.viewpoint_stack[i])
        self.is_loaded = True
  

    #在训练过程中 批量读取图象的代码
    def load_image_to_viewpoint(self,viewpoint_cam):
        image_path = viewpoint_cam.image_fullname  
        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))
 
        bg = np.array([1,1,1]) if self.white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        resized_image_rgb = PILtoTorch(image, image.size)  

         
        image=resized_image_rgb[:3, ...] 

        new_viewpoint_cam= Camera(colmap_id=viewpoint_cam.colmap_id,
                            image_fullname=viewpoint_cam.image_fullname,
                            R=viewpoint_cam.R, 
                            T=viewpoint_cam.T, 
                            FoVx=viewpoint_cam.FoVx, 
                            FoVy=viewpoint_cam.FoVy, 
                            image=image, 
                            gt_alpha_mask=None,
                            image_name=viewpoint_cam.image_name, 
                            uid=viewpoint_cam.uid, 
                            data_device=viewpoint_cam.data_device,
                            keyframe=viewpoint_cam.keyframe)
        return new_viewpoint_cam





def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    #WDD 5-13
    #公共变量卸载
    #用来分配view的缓冲
    image_count =   30  #一次读取进入内存的图象数量 
    viewpoint_buffer_stack=[]
    temp_viewpoint_buffer=None

    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        #if not viewpoint_stack:
        #    viewpoint_stack = scene.getTrainCameras().copy() 
        #viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
   
        #WDD 5-13
        #增加viewpoint的缓冲分配和多线程读取
        #当所有缓冲已经都训练完毕了，重新分配缓冲
        #=======================================================================
        if not viewpoint_buffer_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            index=0
            while viewpoint_stack:
                sub_stack = Viewpoint_Buffer(index,white_background=dataset.white_background)
                for i in range(image_count):
                    if viewpoint_stack:
                        sub_stack.append(viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1)))
                 
                viewpoint_buffer_stack.append(sub_stack)
                index=index+1
 
            #初始的时候读取前俩个buffer的数据
            if viewpoint_buffer_stack: 
                viewpoint_buffer_stack[0].load_images()    
                if len(viewpoint_buffer_stack)>1:
                    viewpoint_buffer_stack[1].load_images()   
                        
        #初始的时候等待 读取
        while not viewpoint_buffer_stack[0].get_is_load() and not temp_viewpoint_buffer:
            # 当前进程暂停1秒
            time.sleep(1)
             

        if viewpoint_buffer_stack[0].get_is_load():
            #获取当前帧
            viewpoint_cam=viewpoint_buffer_stack[0].pop()
        else:
            viewpoint_cam=temp_viewpoint_buffer.pop()

        if viewpoint_buffer_stack[0].is_last(): 
            if len(viewpoint_buffer_stack)>1:
                if viewpoint_buffer_stack[1].is_loaded:
                    viewpoint_buffer_stack.pop(0)
                    if len(viewpoint_buffer_stack)>1:
                        viewpoint_buffer_stack[1].load_images() 
                else:
                    pass
            else:    
                temp_viewpoint_buffer=copy.deepcopy(viewpoint_buffer_stack[0]) 
                viewpoint_buffer_stack.pop(0)

        #=======================================================================


            
            


        #WDD 5-10 XXX
        #为了防止内存溢出，每词都重新读取图象
        #训练会慢很多
        #====================================================================
        is_LotsofImage=True

        #if is_LotsofImage:
        #    viewpoint_cam=load_image_to_viewpoint(viewpoint_cam)
        #====================================================================
        
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        #WDD 5-6 更新当前摄像机对应的关键帧
        #======================================================
        gaussians.current_keyframe=viewpoint_cam.keyframe
        #======================================================

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)


        #WDD 5-2
        #限制运动效果很不好
        #=======================================
        #b = gaussians.indices[:, 1:4]
        #a = gaussians._xyz
        #threshold = 4  # 允许的最大偏差
        # 计算两点之间的差
        #differences = a - b
        #penalties = F.relu(torch.abs(differences) - threshold)
        # 损失是所有超过阈值部分的和
        #Ldis = penalties.sum()
        #========================================


        #loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss =  (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        #=======================================
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            #training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            #WDD =============================================
            #是否进行稠密化，此处为了锁定点数和点的拓扑结构
            
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    
                    min_opacity=0.005 #0.005
                    size_threshold=20
                    gaussians.densify_and_prune(opt.densify_grad_threshold, min_opacity, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        #WDD=============
        #if os.getenv('OAR_JOB_ID'):
        #    unique_str=os.getenv('OAR_JOB_ID')
        #else:
        #    unique_str = str(uuid.uuid4())
        #args.model_path = os.path.join("./output/", unique_str[0:10])
        now = datetime.datetime.now()
        formatted_date = now.strftime("%Y%m%d_%H%M%S")
        args.model_path = os.path.join("./output/", formatted_date)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
 

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    #WDD
    #修改了输入参数，直接f5 运行即可
    #注意修改输入文件的路径
    #==========================================================================================
    #args = parser.parse_args(sys.argv[1:])
    args = parser.parse_args(['-w',
                              '-s','data/head',
                              '-r','1',
                              "--save_iterations",'1','5_000', '10_000','30_000','50_000','100_000',
                              '--iterations','30000', 
                              ])
    #==========================================================================================
    
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

     

    # All done
    print("\nTraining complete.")
