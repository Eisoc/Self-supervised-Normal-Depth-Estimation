from models.baseline import NNET
from utils.utils_edited import *
import os
from models.MotionFusionNet import MotionFusionNet
from torchvision import transforms
import matplotlib.pyplot as plt

flow_path = 'data/imgs/val/flow/test.jpg'
left_image_path = 'data/imgs/val/flow/test.jpg'
flow = Image.open(flow_path)
trans = transforms.ToTensor()
# img = trans(img).to(device).unsqueeze(0)
flow = trans(flow).to(device).unsqueeze(0)

if __name__ == '__main__':
    # NNET Def
    model = NNET()
    model = model.to(device)
    batch_data=model.batch_producer()
    total_size=len(model.test_set)
    print("Length of test set: {}".format(total_size), "Length of 1 test loader: {}".format(len(model.test_loader)))
    color = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 0)]).astype(np.uint8)
    output_path = "./models/test_baseline/outputs"  # 指定输出文件夹
    
    # Motionfusion Net
    model_motion = MotionFusionNet()
    model_motion.load_state_dict(torch.load('checkpoints/checkpoints/best.pt'))
    model_motion = model_motion.to(device)
    model_motion.eval()

    if model.args_geonet.is_train==1:
        model.geonet.train()
    else:
        model.eval()  # 将模型设置为评估模式
        model.geonet.pose_net.eval()
        model.geonet.disp_net.eval()
        # model.geonet.test_depth() # 一次性生成所有图片的深度并保存
        # 下面是读取深度npy文件
        # file_path = model.args_geonet.outputs_dir + os.path.basename(model.args_geonet.ckpt_dir)+ "/rigid__" + str(model.args_geonet.ckpt_index) + '.npy'
        # depth_total = np.memmap(file_path, dtype='float32', mode='r', shape=(total_size, model.args_geonet.img_height, model.args_geonet.img_width))

    with torch.no_grad():
        for i, batch_inputs in enumerate(batch_data):
            model.zero_grad()
            print("-----------Iteration", i,"/", len(model.test_loader)-1, "----------:","total_size=", total_size, "batch_size=", model.args_geonet.batch_size)
            batch_inputs = batch_inputs.to(device)
            
            # 与depth total配套使用
            # pre_depth_ori = depth_total[i:i + model.args_geonet.batch_size]
            # pre_depth = pre_depth_ori.copy()
            # # 将 depth的NumPy 数组转换为 PyTorch 张量
            # pre_depth = torch.from_numpy(pre_depth).to(device)
            
            # 直接调用geonet生成深度
            model.geonet.preprocess_test_data(batch_inputs)
            model.geonet.build_dispnet()
            model.geonet.build_posenet()
            pose_to_csv(model.geonet.poses, "pose.csv")
            pre_depth = model.geonet.depth[0]
            
            norm_pred_final, final_depth= model(pre_depth, batch_inputs)
            
            for j in range(batch_inputs.shape[0]):
                pred_motion = model_motion(batch_inputs[j, :, :, :].float().unsqueeze(0), flow).to('cpu').squeeze(0)
                pred_motion = torch.argmax(pred_motion, dim=0)
                img_label = color[pred_motion]
                img_label = Image.fromarray(np.uint8(img_label))
                img_row = Image.open(left_image_path)
                img = Image.blend(img_row, img_label, 0.3)
                file_path = os.path.join(output_path, f"motion_split_{i*4+j}.png")
                plt.imsave(file_path, img)
            print("Motion_split estimated successfully")
            
            save_tensor_as_image(i, norm_pred_final, "norm_image", output_path)
            save_tensor_as_image(i, final_depth, "depth_image", output_path)
            print("Results saved.")
            torch.cuda.empty_cache()
            # del pre_depth_ori
            del pre_depth
