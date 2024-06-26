from models.baseline import NNET
from utils.utils_edited import *
from copy import deepcopy
from utils.data_readers.frame_utils import *
from utils.utils_raft3d import parse_args_raft3d, make_kitti_in_iterate, folder_builder
import importlib
from models.AutoencoderKL import AutoencoderKL, get_autoencoder

if __name__ == '__main__':
    model = NNET() # NNET Def
    model = model.to(device)
    output_path = "./models/test_baseline/outputs"  # 指定输出文件夹
    args_raft3d = parse_args_raft3d()
    if args_raft3d.headless:
        import matplotlib
        matplotlib.use('Agg') 
    RAFT3D = importlib.import_module(args_raft3d.network).RAFT3D
    model_raft3d = torch.nn.DataParallel(RAFT3D(args_raft3d))
    model_raft3d.load_state_dict(torch.load(args_raft3d.model))
    # model_raft3d = RAFT3D(args_raft3d).to('cuda:1')
    # state_dict = torch.load(args_raft3d.model, map_location='cuda:1')
    # adjusted_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # model_raft3d.load_state_dict(adjusted_state_dict)
   
    if model.args_geonet.is_train==1:
        model.geonet.train()
    else:
        batch_data=model.batch_producer()
        model.eval()  # 将模型设置为评估模式
        model.geonet.pose_net.eval()
        model.geonet.disp_net.eval()
        folder_builder()
        model_raft3d.cuda()
        model_raft3d.eval()
        autoencoder = get_autoencoder()
        
        with torch.no_grad():
            for i, batch_inputs in enumerate(batch_data):
                model.zero_grad()
                print("-----------Iteration", i,"/", len(model.test_loader)-1, "----------:","total_size=", len(model.test_set), "batch_size=", model.args_geonet.batch_size)           
                model.geonet.preprocess_test_data(deepcopy(batch_inputs))
                model.geonet.build_dispnet()
                model.geonet.build_posenet()
                pose_to_csv(model.geonet.poses, output_path+"/pose.csv")
                pre_depth = model.geonet.depth[0] # init depth from geonet
                
                batch_RGB_inputs = batch_inputs[1].to(device)
                norm_pred_final, final_depth= model(pre_depth.clone(), batch_RGB_inputs.clone()) # final D and N
                
                result_track = self.simple_test_track(batch_RGB_inputs.clone())
                
                batch_next_frame = batch_inputs[2][:,3:,:,:].to(device)
                batch_last_frame = batch_inputs[2][:,:3,:,:].to(device)
                save_tensor_as_image(i, norm_pred_final, "norm_image", output_path)
                save_tensor_as_image(i, final_depth, "depth_image", output_path)
                RGBD = torch.cat([batch_RGB_inputs.clone(), final_depth.clone()], dim=1) # torch.Size([4, 4, 128, 416])
                posterior = autoencoder.encode(RGBD)
                # print(posterior.mean.shape) # torch.Size([4, 4, 16, 52])
                # print(posterior.var.shape)
                if i >= 1:
                    make_kitti_in_iterate(model_raft3d, i-1, prev_rgb, batch_RGB_inputs, prev_depth, final_depth, prev_intrinsics) # model, i_batch, image1, image2, depth1, depth2, intrinsics
                print("Results saved.")
                torch.cuda.empty_cache()
                prev_rgb, prev_depth, prev_intrinsics = batch_RGB_inputs, final_depth, batch_inputs[0].to(device)
                del pre_depth
