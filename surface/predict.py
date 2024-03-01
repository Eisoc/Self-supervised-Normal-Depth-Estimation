from models.baseline import NNET
from utils.utils_edited import *
import os

if __name__ == '__main__':
    total_size = 4877
    model = NNET()
    model = model.to(device)
    batch_data=model.batch_producer()
    if model.args_geonet.is_train==1:
        model.geonet.train()
    elif model.args_geonet.is_train==2:
        pre_depth = model.geonet.test_depth()
    else:
        file_path = model.args_geonet.outputs_dir + os.path.basename(model.args_geonet.ckpt_dir)+ "/rigid__" + str(model.args_geonet.ckpt_index) + '.npy'
        # pre_depth = np.load(file_path)
        # pre_depth = torch.from_numpy(pre_depth).to(device)
        # model.geonet.test_depth()
        depth_total = np.memmap(file_path, dtype='float32', mode='r', shape=(total_size, 128, 416))
        # pre_depth = model.geonet.test_depth() 
    
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():
        for i, batch_inputs in enumerate(batch_data):
            model.zero_grad()
            print("--------------Iteration---------------:", i, "total_size=", total_size, "batch_size=", model.args_geonet.batch_size)
            batch_inputs = batch_inputs.to(device)
            pre_depth_ori = depth_total[i:i + model.args_geonet.batch_size]
            pre_depth = pre_depth_ori.copy()
            # 将 NumPy 数组转换为 PyTorch 张量
            pre_depth = torch.from_numpy(pre_depth).to(device)
            norm_pred_final, final_depth= model(pre_depth, batch_inputs)
            # print(norm_pred_final.size(), final_depth.size())
            output_path = "./models/test_baseline/outputs"  # 指定输出文件夹
            save_tensor_as_image(i, norm_pred_final, "norm_image", output_path)
            save_tensor_as_image(i, final_depth, "depth_image", output_path)
            del pre_depth
            del pre_depth_ori
            torch.cuda.empty_cache()
            # print(torch.cuda.memory_summary(device=None, abbreviated=False))
    # print(out.shape)