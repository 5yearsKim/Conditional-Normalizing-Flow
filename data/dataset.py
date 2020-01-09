import os
import random
import torch
import torch.utils.data
from torchvision import transforms
from PIL import Image
from skimage import feature

def files_to_list(filename):
    """
    Takes a text file of filenames and makes a list of filenames
    """
    with open(filename, encoding='utf-8') as f:
        files = f.readlines()

    files = [f.rstrip() for f in files]
    return files

class ImgDatasets(torch.utils.data.Dataset):
    def __init__(self, root_dir, files, mode='sketch'):
        self.img_files = files_to_list(files)
        self.root_dir = root_dir
        self.origin_transform = transforms.Compose([
            transforms.CenterCrop(128),
            transforms.Resize(64),
            transforms.RandomHorizontalFlip()
        ])
        self.gray_transform = transforms.Compose([
           transforms.Grayscale()
        ])
        self.ToTensor = transforms.ToTensor()
        random.seed(1234)
        random.shuffle(self.img_files)
        self.mode = mode

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.img_files[index])
        image = Image.open(img_name)
        image = self.origin_transform(image)
        gray_img = self.gray_transform(image)
        image = self.ToTensor(image)
        gray_img = self.ToTensor(gray_img)
        if self.mode == 'gray':
            return (image, gray_img)
        else:
            edges = feature.canny(gray_img.squeeze(0).numpy(), sigma=0.3)
            edges = torch.from_numpy(edges).type(torch.float)
            edges = edges.unsqueeze(0)
            return (image, edges)





if __name__ == "__main__":

    filename = "../train_files.txt"
    att_file = "./list_attr_celeba.txt"
    sample_size = 16
    dataset = ImgDatasets("celeba_sample", filename, att_file)
    loader = torch.utils.data.DataLoader(dataset, batch_size=sample_size)
    original, cond_img = next(iter(loader))
    data = { "original": original,
             "cond_img": cond_img,
    }
    torch.save(data, "../inference_data/for_inference.pt")
