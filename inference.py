import torch
import os
import torchvision
from models import Glow

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.ckpt_path)
    ckpt_args = ckpt["args"]
    net = Glow(num_channels=ckpt_args.num_channels,
               num_levels=ckpt_args.num_levels,
               num_steps=ckpt_args.num_steps,
               img_size=ckpt_args.img_size,
               dec_size=ckpt_args.dec_size).to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, ckpt_args.gpu_ids)
    net.load_state_dict(ckpt['net'])

    cond_data = torch.load(args.cond_data)
    original, cond_img = cond_data["original"], cond_data["cond_img"].to(device)


    # style transfer
    synth_img, target = style_transfer(net, original, cond_img, target_index=args.index)

    ######3#
    os.makedirs('inference_data', exist_ok=True)
    origin_concat = torchvision.utils.make_grid(original, nrow=4, padding=2, pad_value=255)
    img_concat = torchvision.utils.make_grid(synth_img, nrow=4, padding=2, pad_value=255)
    torchvision.utils.save_image(origin_concat, args.output_dir + 'original.png')
    torchvision.utils.save_image(img_concat, args.output_dir + '/synthesized.png')
    torchvision.utils.save_image(target, args.output_dir + 'cond_img.png')

@torch.no_grad()
def sample(net, encoder, dec_img, extra_cond, img_size=64):
    z = encoder(extra_cond, sigma=0.1)
    x, _ = net(z, dec_img, reverse=True)
    x = torch.sigmoid(x)
    return x

@torch.no_grad()
def random_sample(net, dec_img, extra_cond, device, sigma=1., img_size=64):
    B, C, H, W = dec_img.shape
    z = torch.randn(B, 3, img_size, img_size).to(device) * sigma
    x, _ = net(z, dec_img, reverse=True)
    x = torch.sigmoid(x)
    return x

@torch.no_grad()
def style_transfer(net, original, cond_img, target_index=0, img_size=64):
    B = original.size(0)
    target = [cond_img[target_index] for _ in range(B)]
    target = torch.stack(target)
    z, _ = net(original, cond_img)
    reconstructed, _ = net(z, target, reverse=True)
    reconstructed = torch.sigmoid(reconstructed)
    return reconstructed, cond_img[target_index]


def make_data_for_interpolation(index, decimated, extra_cond, num_samples=8, feature = "Smiling"):
    f_map = {"Young":0, "Smiling":1, "Pale_Skin":2 }
    f_i = f_map[feature]
    decimated = [decimated[index] for _ in range(num_samples)]
    decimated = torch.stack(decimated)
    extra_cond = [extra_cond[index] for _ in range(num_samples)]
    extra_cond = torch.stack(extra_cond, dim=0)
    for i in range(num_samples):
        extra_cond[i][f_i] = -1. + 2*i/(num_samples-1)
    return (decimated, extra_cond)



if __name__ == "__main__":
    import argparse

    def str2bool(s):
        return s.lower().startswith('t')

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--filelist_path")
    parser.add_argument('-p', '--ckpt_path', default="ckpts/best.pth.tar")
    parser.add_argument('-c', '--cond_data', default="inference_data/for_inference.pt")
    parser.add_argument('-o', "--output_dir", default="inference_data/")
    parser.add_argument("-s", "--sigma", default=1.0, type=float)
    parser.add_argument('-n', "--num_samples", default=8)
    parser.add_argument('-t', "--feature", default="Smiling")
    parser.add_argument('-i', "--index", default=1)

    args = parser.parse_args()

    main(args)
