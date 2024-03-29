from models.baseline import NNET
from utils.utils_edited import *
import os
from models.MotionFusionNet import MotionFusionNet
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.models.optical_flow import Raft_Large_Weights
from torchvision.models.optical_flow import raft_large
from copy import deepcopy

if __name__ == '__main__':
    
    model = NNET() # NNET Def
    model = model.to(device)
    color = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 0)]).astype(np.uint8)
    output_path = "./models/test_baseline/outputs"  # 指定输出文件夹
    
    model_motion = MotionFusionNet()  # Motionfusion Net
    model_motion.load_state_dict(torch.load('checkpoints/checkpoints/best.pt'))
    model_motion = model_motion.to(device)
    model_motion.eval()
    
    model_RAFT = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
   

    if model.args_geonet.is_train==1:
        model.geonet.train()
    else:
        batch_data=model.batch_producer()
        model.eval()  # 将模型设置为评估模式
        model.geonet.pose_net.eval()
        model.geonet.disp_net.eval()
        model_RAFT.eval()
        # model.geonet.test_depth() # 一次性生成所有图片的深度并保存
        # 下面是读取深度npy文件
        # file_path = model.args_geonet.outputs_dir + os.path.basename(model.args_geonet.ckpt_dir)+ "/rigid__" + str(model.args_geonet.ckpt_index) + '.npy'
        # depth_total = np.memmap(file_path, dtype='float32', mode='r', shape=(total_size, model.args_geonet.img_height, model.args_geonet.img_width))

        with torch.no_grad():
            for i, batch_inputs in enumerate(batch_data):
                model.zero_grad()
                print("-----------Iteration", i,"/", len(model.test_loader)-1, "----------:","total_size=", len(model.test_set), "batch_size=", model.args_geonet.batch_size)
                # 与depth total配套使用
                # pre_depth_ori = depth_total[i:i + model.args_geonet.batch_size]
                # pre_depth = pre_depth_ori.copy()
                # # 将 depth的NumPy 数组转换为 PyTorch 张量
                # pre_depth = torch.from_numpy(pre_depth).to(device)
                
                model.geonet.preprocess_test_data(deepcopy(batch_inputs))
                model.geonet.build_dispnet()
                model.geonet.build_posenet()
                pose_to_csv(model.geonet.poses, output_path+"/pose.csv")
                pre_depth = model.geonet.depth[0] # init depth from geonet
                
                batch_RGB_inputs = batch_inputs[0].to(device)
                norm_pred_final, final_depth= model(pre_depth.clone(), batch_RGB_inputs.clone()) # final D and N
                
                batch_next_frame = batch_inputs[1][:,3:,:,:].to(device)
                batch_last_frame = batch_inputs[1][:,:3,:,:].to(device)
                list_of_flows = model_RAFT(batch_RGB_inputs.clone().float() / 255.0, batch_next_frame.clone().float() / 255.0)
                predicted_flows = list_of_flows[-1] # optical flow
                print("Optical flow estimated successfully.")
                
                for j in range(batch_RGB_inputs.shape[0]): # for motion_split
                    # print(batch_inputs.size(),flow.size())
                    pred_motion = model_motion(batch_RGB_inputs[j, :, :, :].float().unsqueeze(0), convert_flow_dim(predicted_flows[j, :, :, :])).to('cpu').squeeze(0)
                    pred_motion = torch.argmax(pred_motion, dim=0)
                    img_label = color[pred_motion]
                    img_label = Image.fromarray(np.uint8(img_label))
                    img_row = batch_last_frame[j, :, :, :].squeeze(0)             
                    img_row_np = np.transpose(img_row.cpu().detach().numpy(), (1, 2, 0))  # 从CHW转换到HWC
                    img_row_pil = Image.fromarray((img_row_np * 255).astype(np.uint8))
                    img = Image.blend(img_row_pil, img_label, 0.3)
                    file_path = os.path.join(output_path, f"motion_split_{i*4+j}.png")
                    plt.imsave(file_path, img)
                print("Motion_split estimated successfully")

                save_tensor_as_image(i, norm_pred_final, "norm_image", output_path)
                save_tensor_as_image(i, final_depth, "depth_image", output_path)
                save_tensor_as_image(i, predicted_flows, "optical_flow", output_path)
                print("Results saved.")
                torch.cuda.empty_cache()
                # del pre_depth_ori
                del pre_depth
