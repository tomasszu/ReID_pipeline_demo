import numpy as np 
import torch
from torchvision import transforms
from PIL import Image
import cv2

import torch.nn as nn


from vehicle_reid.load_model import load_model_from_opts

class ExtractingFeatures:
    """A class to handle feature extraction from cropped vehicle images using a pre-trained model.
    This class initializes the model, applies necessary transformations to the images, and extracts feature vectors.
    It supports batch processing of images and can handle flipping images for data augmentation.
    Attributes:
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').
        model (nn.Module): The pre-trained model for feature extraction.
        data_transforms (transforms.Compose): A series of transformations to apply to the images before feature extraction.
    Methods:
        __init__(device):
            Initializes the ExtractingFeatures with the specified device and loads the pre-trained model.
        fliplr(img):
            Flips images horizontally in a batch.
        extract_feature(model, X):
            Extracts embeddings of a batch of image tensors X.
            X should be of shape [B, C, H, W].
        get_features_batch(obj_crop_list):
            Accepts a list of (obj_id, crop) tuples and returns a list of (obj_id, feature_vector) tuples.
        get_feature(image):
            Accepts a single image and returns its feature vector.
    """
    def __init__(self, device):

        self.device = device
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

    def extract_feature(self, model, X):
        """
        Extract embeddings of a batch of image tensors X.
        X should be of shape [B, C, H, W]
        """
        X = X.to(self.device)

        with torch.no_grad():
            features = model(X)  # shape: [B, D]
            X_flipped = self.fliplr(X)
            features_flipped = model(X_flipped)

        features = features + features_flipped
        fnorm = torch.norm(features, p=2, dim=1, keepdim=True)
        features = features.div(fnorm)  # shape: [B, D]

        return features
    
    def get_features_batch(self, obj_crop_list):
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
        ]).to(self.device)

        features = self.extract_feature(self.model, processed_images)
        features = features.detach().cpu().numpy()

        return list(zip(obj_ids, features))

    
    def get_feature(self, image):
        """Accepts a single image and returns its feature vector.
        Args:
            image (np.ndarray): The input image to extract features from.
        Returns:
            np.ndarray: The feature vector extracted from the image.
        """

        image = [image]

        X_images = torch.stack(tuple(map(self.data_transforms, image))).to(self.device)

        features = [self.extract_feature(self.model, X_images)]
        features = torch.stack(features).detach().cpu()

        features_array = np.array(features)

        return features_array

