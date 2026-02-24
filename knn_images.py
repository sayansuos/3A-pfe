import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image


def get_similarities(
    input_image_path,
    database_folder,
    vqvae_model,
    device,
    img_size=128,
    metric="cosine",
):

    vqvae_model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    input_img = (
        transform(Image.open(input_image_path).convert("RGB")).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        input_embedding = vqvae_model._pre_vq_conv(vqvae_model._encoder(input_img))
        input_embedding = input_embedding.view(1, -1)

    results = []

    for img_name in os.listdir(database_folder):
        if img_name.endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(database_folder, img_name)

            target_img = (
                transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            )
            with torch.no_grad():
                target_emb = vqvae_model._pre_vq_conv(vqvae_model._encoder(target_img))
                target_emb = target_emb.view(1, -1)

            if metric == "cosine":
                score = F.cosine_similarity(input_embedding, target_emb).item()

            if metric == "euclidian":
                score = F.pairwise_distance(input_embedding, target_emb, p=2).item()

            results.append((path, score))

    if metric == "cosine":
        results.sort(key=lambda x: x[1], reverse=True)

    if metric == "euclidian":
        results.sort(key=lambda x: x[1], reverse=False)

    return results


def get_nearest_img(
    input_image_path,
    database_folder,
    vqvae_model,
    device,
    img_size=128,
    k=5,
    metric="cosine",
):
    similarities = get_similarities(
        input_image_path, database_folder, vqvae_model, device, img_size, metric
    )
    top_k = similarities[:k]

    best_scores = [res[1] for res in top_k]
    best_paths = [res[0] for res in top_k]

    plt.figure(figsize=(20, 10))

    plt.subplot(1, len(best_paths) + 1, 1)
    plt.imshow(Image.open(input_image_path))
    plt.title(f"Query\n{os.path.basename(input_image_path)}", color="red")
    plt.axis("off")

    for i in range(len(best_paths)):
        plt.subplot(1, len(best_paths) + 1, i + 2)
        plt.imshow(Image.open(best_paths[i]))
        if metric == "cosine":
            plt.title(
                f"Cosine Similarity Score: {best_scores[i]:.4f}\n{os.path.basename(best_paths[i])}"
            )
        if metric == "euclidian":
            plt.title(
                f"Euclidian Distance: {best_scores[i]:.4f}\n{os.path.basename(best_paths[i])}"
            )
        plt.axis("off")
    plt.show()

    return best_scores, best_paths
