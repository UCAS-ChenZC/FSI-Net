from argparse import ArgumentParser
import torch
from models.evaluator import *
import numpy as np
from matplotlib import pyplot as plt
print(torch.cuda.is_available())

def process_tensor(tens):
    # print(f"Type of tens: {type(tens)}")  # 调试代码
    # img = tens[0].cpu()
    img = tens[0].cpu()
    # print(f"Type after .cpu(): {type(img)}")  # 进一步检查
    
    img = torch.mean(img,dim=1)
    img = torch.squeeze(img)
    img = img.numpy()

    img = (img - img.min())/(img.max()-img.min())
    return img
"""
eval the CD model
"""

def main():
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--project_name', default='511_noMapper_ohem_lr0.0002(0)', type=str)
    parser.add_argument('--print_models', default=False, type=bool, help='print models')
    parser.add_argument('--checkpoints_root', default='checkpoints', type=str)
    parser.add_argument('--vis_root', default='vis', type=str)

    # data
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dataset', default='CDDataset', type=str)
    parser.add_argument('--data_name', default='heat_map', type=str)            #LEVIR

    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--split', default="test", type=str)

    parser.add_argument('--img_size', default=256, type=int)

    # model
    parser.add_argument('--n_class', default=2, type=int)
    parser.add_argument('--embed_dim', default=256, type=int)
    parser.add_argument('--net_G', default='SwinChangeFormer', type=str,
                        help='base_resnet18 | base_transformer_pos_s4_dd8 | base_transformer_pos_s4_dd8_dedim8|')

    parser.add_argument('--checkpoint_name', default='best_ckpt.pt', type=str)

    args = parser.parse_args()
    utils.get_device(args)
    print(args.gpu_ids)

    #  checkpoints dir
    args.checkpoint_dir = os.path.join(args.checkpoints_root, args.project_name)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    #  visualize dir
    args.vis_dir = os.path.join(args.vis_root, args.project_name)
    os.makedirs(args.vis_dir, exist_ok=True)
    def custom_repr(self):
        return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr
    dataloader = utils.get_loader(args.data_name, img_size=args.img_size,
                                  batch_size=args.batch_size, is_train=False,
                                  split=args.split)
    model = CDEvaluator(args=args, dataloader=dataloader)

    model.eval_models(checkpoint_name=args.checkpoint_name)

    x1 = model.net_G.x1
    x2 = model.net_G.x2
    # x4 = model.net_G.TDec_x2_V2.x44
    # x4 = process_tensor(x4)
    x1,x2 = process_tensor(x1),process_tensor(x2)
   
    plt.imshow(x2,cmap='jet_r')       #hot -> 低值变黑，高值变红黄   coolwarm ->  低值变蓝，高值变红  默认 viridis（黄绿蓝）  jet（经典风格）：适合 科学计算可视化（蓝绿黄红）
    plt.colorbar()
    plt.show()
    print(x2)
if __name__ == '__main__':
    main()

