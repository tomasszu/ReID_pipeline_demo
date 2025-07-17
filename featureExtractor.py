import numpy as np 
import torch
from torchvision import transforms
from PIL import Image
import cv2

import torch.nn as nn


from vehicle_reid.load_model import load_model_from_opts

class ExtractingFeatures:
    def __init__(self):

        self.device = "cuda"
        self.model = load_model_from_opts("vehicle_reid/models/veri+vehixlex_editTrainPar1/opts.yaml", 
                                     ckpt="vehicle_reid/models/veri+vehixlex_editTrainPar1/net_39.pth", 
                                     remove_classifier=True)
        self.model.eval()
        self.model.to(self.device)

        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def fliplr(self, img):
        """flip images horizontally in a batch"""
        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
        inv_idx = inv_idx.to(img.device)
        img_flip = img.index_select(3, inv_idx)
        return img_flip

    def extract_feature(self, model, X, device="cuda"):
        """
        Extract embeddings of a batch of image tensors X.
        X should be of shape [B, C, H, W]
        """
        X = X.to(device)

        with torch.no_grad():
            features = model(X)  # shape: [B, D]
            X_flipped = self.fliplr(X)
            features_flipped = model(X_flipped)

        features = features + features_flipped
        fnorm = torch.norm(features, p=2, dim=1, keepdim=True)
        features = features.div(fnorm)  # shape: [B, D]

        return features
    
    def get_features_batch(self, obj_crop_list, device="cuda"):
        """
        Accepts list of (obj_id, crop) tuples.
        Returns list of (obj_id, feature_vector) tuples.
        """
        if len(obj_crop_list) == 0:
            return []

        obj_ids, crops = zip(*obj_crop_list)  # unzip
        processed_images = torch.stack([
            self.data_transforms(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            for img in crops
        ]).to(device)

        features = self.extract_feature(self.model, processed_images, device=device)
        features = features.detach().cpu().numpy()

        return list(zip(obj_ids, features))

    
    def get_feature(self, image, device="cuda"):

        image = [image]

        X_images = torch.stack(tuple(map(self.data_transforms, image))).to(device)

        features = [self.extract_feature(self.model, X_images)]
        features = torch.stack(features).detach().cpu()

        features_array = np.array(features)

        return features_array

