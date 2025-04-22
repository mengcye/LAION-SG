import os
import torch
import numpy as np
import time
from tqdm import tqdm
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
from torchvision.transforms import Resize

device = "cuda" if torch.cuda.is_available() else "cpu"
resize_transform = Resize((256, 256), antialias=False)

all_npz_dirs = [f"npz_files_{i}" for i in range(8)]

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

fid_results_file = os.path.join(output_dir, "fid_results.txt")


def calculate_activation_statistics_from_tensor_incremental(images_tensor, model, device='cpu', batch_size=50):
    model.eval()
    n_samples = 0
    sum_activations = None
    sum_sq_activations = None
    with torch.no_grad():
        for i in range(0, images_tensor.shape[0], batch_size):
            batch = images_tensor[i:i + batch_size].to(device)
            pred = model(batch)[0]
            pred = pred.squeeze(3).squeeze(2).cpu().numpy().astype(np.float64)  # Shape: (batch_size, 2048)
            if sum_activations is None:
                sum_activations = np.sum(pred, axis=0)
                sum_sq_activations = np.dot(pred.T, pred)
            else:
                sum_activations += np.sum(pred, axis=0)
                sum_sq_activations += np.dot(pred.T, pred)
            n_samples += pred.shape[0]
    return sum_activations, sum_sq_activations, n_samples


if __name__ == '__main__':
    start_time = time.time()

    print(f"Processing directories: {all_npz_dirs}")

    total_sum_real, total_sum_sq_real, total_n_real = None, None, 0
    total_sum_fake, total_sum_sq_fake, total_n_fake = None, None, 0

    # Initialize InceptionV3 model
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model_fid = InceptionV3([block_idx]).to(device)
    model_fid.eval()

    # Counters for processed files
    processed_files = 0
    skipped_files = 0

    for npz_dir in all_npz_dirs:
        npz_files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')]
        for npz_file in tqdm(npz_files, desc=f'Processing files in {npz_dir}'):
            try:
                data = np.load(npz_file)
                real_images = data['real_images'].astype(np.float32) / 255.0  # Shape: (batch_size, H, W, C)
                fake_images = data['fake_images'].astype(np.float32) / 255.0

                real_images_tensor = torch.from_numpy(real_images).permute(0, 3, 1, 2)
                fake_images_tensor = torch.from_numpy(fake_images).permute(0, 3, 1, 2)

                real_images_tensor = resize_transform(real_images_tensor)
                fake_images_tensor = resize_transform(fake_images_tensor)

                # Normalize to [-1, 1] range for both real and fake images
                real_images_tensor = real_images_tensor * 2 - 1
                fake_images_tensor = fake_images_tensor * 2 - 1  # Also normalize fake images

                sum_real, sum_sq_real, n_real = calculate_activation_statistics_from_tensor_incremental(
                    real_images_tensor, model_fid, device=device, batch_size=50
                )
                if total_sum_real is None:
                    total_sum_real, total_sum_sq_real = sum_real, sum_sq_real
                else:
                    total_sum_real += sum_real
                    total_sum_sq_real += sum_sq_real
                total_n_real += n_real

                sum_fake, sum_sq_fake, n_fake = calculate_activation_statistics_from_tensor_incremental(
                    fake_images_tensor, model_fid, device=device, batch_size=50
                )
                if total_sum_fake is None:
                    total_sum_fake, total_sum_sq_fake = sum_fake, sum_sq_fake
                else:
                    total_sum_fake += sum_fake
                    total_sum_sq_fake += sum_sq_fake
                total_n_fake += n_fake

                processed_files += 1

                # Free memory
                del data, real_images, fake_images, real_images_tensor, fake_images_tensor
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error processing file {npz_file}: {str(e)}")
                skipped_files += 1
                continue

    # Calculate mean and covariance for real images
    mu_real = total_sum_real / total_n_real
    sigma_real = (total_sum_sq_real - total_n_real * np.outer(mu_real, mu_real)) / (total_n_real - 1)

    # Calculate mean and covariance for fake images
    mu_fake = total_sum_fake / total_n_fake
    sigma_fake = (total_sum_sq_fake - total_n_fake * np.outer(mu_fake, mu_fake)) / (total_n_fake - 1)

    # Calculate FID
    fid_value = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time

    # Prepare result message
    result_message = f"FID Results\n"
    result_message += f"==========\n\n"
    result_message += f"FID value: {fid_value:.4f}\n\n"
    result_message += f"Statistics:\n"
    result_message += f"- Total files processed: {processed_files}\n"
    result_message += f"- Files skipped due to errors: {skipped_files}\n"
    result_message += f"- Total real images: {total_n_real}\n"
    result_message += f"- Total fake images: {total_n_fake}\n"
    result_message += f"- Processing time: {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)\n"

    # Print and save results
    print(result_message)

    # Write results to file
    with open(fid_results_file, "w") as f:
        f.write(result_message)

    print(f"Results saved to {fid_results_file}")