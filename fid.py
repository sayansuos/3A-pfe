import scipy
import numpy as np
from scipy.linalg import sqrtm

import torch
import torch.nn as nn
from torchvision import models, transforms

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.mps.is_available() else "cpu"
)
inception_model = models.inception_v3(weights="IMAGENET1K_V1")
inception_model.fc = nn.Identity()
inception_model.eval().to(device)


def get_inception_features(batch_tensor):
    preprocess = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    batch = preprocess(batch_tensor)
    with torch.no_grad():
        features = inception_model(batch)
    return features.detach().cpu().numpy()


def calculate_fid(real_embeddings, generated_embeddings):
    mu1, sigma1 = real_embeddings.mean(axis=0), np.cov(real_embeddings, rowvar=False)
    mu2, sigma2 = generated_embeddings.mean(axis=0), np.cov(
        generated_embeddings, rowvar=False
    )
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid
