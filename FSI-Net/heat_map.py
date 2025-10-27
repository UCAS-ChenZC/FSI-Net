import cv2
import time
import os
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from functools import partial
from models.ChangeFormer import SwinChangeFormer
savepath1 = '/home/solid/CD/datasets/LEVIR-CD256(2)/heatmap/000/'
savepath2 = '/home/solid/CD/datasets/LEVIR-CD256(2)/heatmap/111/'
if not os.path.exists(savepath1) and not os.path.exists(savepath2):
    os.mkdir(savepath1)
    os.mkdir(savepath2)


def draw_features(channel, x, savename, scale):
    for i in range(channel):
        img = x[0, i, :, :]
        pmin = np.min(img)
        pmax = np.max(img)
        img = ((img - pmin) / (pmax - pmin + 0.000001)) * 255  # float在[0，1]之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 注意此处的三通道热力图是cv2专有的GBR排列
        # img = img[:, :, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        cv2.imshow('{}.jpg'.format(i), img)
        cv2.waitKey(10)
        cv2.destroyAllWindows()
        cv2.imwrite(savename + "{}.jpg".format(i), img)
        print("{}/{}".format(i, channel))




# class ft_net(nn.Module):

#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.features = []
#         # 用于跟踪每个DAFM模块被调用的次数
#         self.dafm_call_count = {name: 0 for name in ["DAFM0", "DAFM1", "DAFM2", "DAFM3", "DAFM4"]}

class ft_net(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.features = []
        self.branch_marker = {"current": "x1"}  # 标记当前处理的分支

        # 注册钩子到每个Swin阶段
        for stage_idx, stage in enumerate(self.model.swin.stages):  # 假设阶段存储在stages中
            stage.register_forward_hook(
                partial(self.hook_swin_stage, stage_idx=stage_idx)
            )
        self.dafm_call_count = {name: 0 for name in ["DAFM0", "DAFM1", "DAFM2", "DAFM3", "DAFM4"]}
    def hook_swin_stage(self, module, input, output, stage_idx):
        branch = self.branch_marker["current"]
        for scale_idx, feat in enumerate(output):
            self.features.append((
                f"swin_stage{stage_idx}_scale{scale_idx}_{branch}",
                feat
            ))

    def forward(self, x1, x2):
        # 处理x1分支
        self.branch_marker["current"] = "x1"
        x1 = self.model.swin(x1)

        # 处理x2分支
        self.branch_marker["current"] = "x2"
        x2 = self.model.swin(x2)
        # 前向传播
        return self.model(x1, x2)

model = SwinChangeFormer(output_nc=2,embed_dim=256).cuda()
checkpoint = torch.load("checkpoints/20250304/test_0304_FFM(5)/best_ckpt.pt")
model.load_state_dict(checkpoint["model_G_state_dict"])
model.eval()
ft_model = ft_net(model)
img1 = cv2.imread('/home/solid/CD/datasets/LEVIR-CD256(2)/LEVIR-CD256/A/test_1_2.png')
img1 = cv2.resize(img1, (256, 256))
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('/home/solid/CD/datasets/LEVIR-CD256(2)/LEVIR-CD256/B/test_1_2.png')
img2 = cv2.resize(img2, (256, 256))
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
img1 = transform(img1).cuda()
img1 = img1.unsqueeze(0)
img2 = transform(img2).cuda()
img2 = img2.unsqueeze(0)

with torch.no_grad():
    out = ft_model(img1, img2)
    
    for name, feat in ft_model.features:
        # 解析特征名称（示例："swin_stage0_scale0_x1"）
        parts = name.split('_')
        module_type = parts[0]    # "swin"
        stage_idx = parts[1].replace("stage", "")  # "0"
        scale_idx = parts[2].replace("scale", "")  # "0"
        branch = parts[3]        # "x1"或"x2"

        # 创建保存目录
        save_root = savepath1 if branch == "x1" else savepath2
        save_dir = os.path.join(
            save_root,
            f"{module_type}/stage{stage_idx}/scale{scale_idx}"
        )
        os.makedirs(save_dir, exist_ok=True)

        # 提取特征数据
        feat_np = feat.cpu().numpy()
        channels = feat_np.shape[1]

        # 生成热力图
        draw_features(
            channel=channels,
            x=feat_np,
            savename=os.path.join(save_dir, "channel_"),
            scale=4  # 根据实际特征尺寸调整
        )
