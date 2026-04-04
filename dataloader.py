## dataset loader which loads images, masks and ground truth 3d meshes ##

import os
import json
import numpy as np
from PIL import Image
import trimesh
import torch
from torch.utils.data import Dataset
import cv2

class Pix3DChairDataset(Dataset):
    def __init__(self, root_dir, split='test', img_size=224):
        """
        Args:
        
        :param root_dir: Path to pix3d dataset (/home/niranjan/2d-image-to-3D-model-generation/data/pix3d)
        :param split: 'train', 'test', or 'val'
        :param img_size: Size to resize images to
        """

        self.root_dir = root_dir
        self.img_size = img_size
        self.split = split

        #loading metadata
        metadata_path = os.path.join(root_dir, 'pix3d.json')
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        #filter for chair data subset
        self.chair_data = [item for item in self.metadata if item ['category'] == 'chair']

        print(f"Loaded {len(self.chair_data)} chair samples from Pix3D")

    def __len__(self):
        return len(self.chair_data)
    
    def __getitem__(self, idx):
        item = self.chair_data[idx]

        #loading image
        img_path = os.path.join(self.root_dir, item['img'])
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.img_size, self.img_size))
        image = np.array(image).astype(np.float32) / 255.0

        #loading mask
        mask_path = os.path.join(self.root_dir, item['mask'])
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize((self.img_size, self.img_size))
        mask = np.array(mask).astype(np.float32) / 255.0

        #applying mask to image
        image = image * mask[:, :, np.newaxis]

        #loading ground truth mesh
        model_path = os.path.join(self.root_dir, item['model'])
        try:
            gt_mesh = trimesh.load(model_path, force='mesh', process = False)
            #making sure the object is a trimesh object and not a scene
            if isinstance(gt_mesh, trimesh.Scene):
                #extracting the first geometry from the scene
                gt_mesh = list(gt_mesh.geometry.values())[0]

        except Exception as e:
            print(f"Warning: Could not load mesh for this sample {idx}: {e}")
            gt_mesh = None

        #converting to torch tensors
        image_tensor = torch.from_numpy(image).permute(2,0,1) #C X H X W
        mask_tensor = torch.from_numpy(mask).unsqueeze(0) # 1 x H x W

        return {
            'image': image_tensor,
            'mask': mask_tensor,
            'gt_mesh': gt_mesh,
            'img_path': img_path,
            'model_path': model_path,
            'idx': idx
        }
    
    def get_sample_by_index(self, idx):
        """
        Getting a specific sample by index
        """
        return self.__getitem__(idx)
    
def test_dataloader():
    """
    Testing the dataloader
    """
    dataset = Pix3DChairDataset(
        root_dir='/home/niranjan/2D-image-to-3D-model-generation/data/pix3d'
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    #test loading first sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Has GT mesh: {sample['gt_mesh'] is not None}")
    if sample['gt_mesh'] is not None:
        print(f"  GT mesh type: {type(sample['gt_mesh'])}")
        print(f"  GT mesh vertices: {len(sample['gt_mesh'].vertices)}")
        print(f"  GT mesh faces: {len(sample['gt_mesh'].faces)}")
    
    return dataset


if __name__ == '__main__':
    test_dataloader()