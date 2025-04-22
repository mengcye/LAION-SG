import os
import torch
import clip
import numpy as np
import time
from torchvision.transforms import ToPILImage
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

all_npz_dirs = [f"npz_files_{i}" for i in range(8)]

print(all_npz_dirs)

output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

clip_scores_file = os.path.join(output_dir, "clip_scores.txt")
summary_file = os.path.join(output_dir, "clip_score_summary.txt")

clip_scores = []

with open(clip_scores_file, "w") as f_output:
    start_time = time.time()

    for npz_dir in all_npz_dirs:
        npz_files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith('.npz')]
        for npz_file in tqdm(npz_files, desc=f'Processing files in {npz_dir}'):
            try:
                data = np.load(npz_file)
                real_images = data['real_images'].astype(np.float32) / 255.0
                fake_images = data['fake_images'].astype(np.float32) / 255.0

                real_images_tensor = torch.from_numpy(real_images).permute(0, 3, 1, 2)
                fake_images_tensor = torch.from_numpy(fake_images).permute(0, 3, 1, 2)

                pil_real_image = ToPILImage()(real_images_tensor[0])
                pil_fake_image = ToPILImage()(fake_images_tensor[0])

                image1 = preprocess(pil_real_image).unsqueeze(0).to(device)
                image2 = preprocess(pil_fake_image).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_features1 = model.encode_image(image1)
                    image_features2 = model.encode_image(image2)

                image_features1 /= image_features1.norm(dim=-1, keepdim=True)
                image_features2 /= image_features2.norm(dim=-1, keepdim=True)
                similarity = (image_features1 @ image_features2.T).item()

                clip_scores.append(similarity)

                # Write individual score to the file
                f_output.write(f"File: {npz_file}, CLIP Score: {similarity:.4f}\n")

                current_time = time.time()
                elapsed_time = current_time - start_time
            except Exception as e:
                print(f"File {npz_file} is not a valid .npz file or has other issues. Skipping. Error: {str(e)}")
                continue  # Skip current file, continue with next file

    # Calculate mean CLIP score
    mean_clip_score = np.mean(clip_scores)
    std_clip_score = np.std(clip_scores)
    final_message = f'CLIP Score mean between SD originals and GT: {mean_clip_score:.4f}\n'
    final_message += f'CLIP Score standard deviation: {std_clip_score:.4f}\n'
    final_message += f'Total files processed: {len(clip_scores)}\n'
    final_message += f'Total runtime: {time.time() - start_time:.2f} seconds'

    print(final_message)
    f_output.write("\n" + final_message + "\n")

# Write summary to a separate file
with open(summary_file, "w") as f_summary:
    f_summary.write(f'CLIP Score Summary\n')
    f_summary.write(f'=================\n\n')
    f_summary.write(f'Mean CLIP Score: {mean_clip_score:.4f}\n')
    f_summary.write(f'Standard Deviation: {std_clip_score:.4f}\n')
    f_summary.write(f'Minimum Score: {min(clip_scores):.4f}\n')
    f_summary.write(f'Maximum Score: {max(clip_scores):.4f}\n')
    f_summary.write(f'Total files processed: {len(clip_scores)}\n')
    f_summary.write(f'Total runtime: {time.time() - start_time:.2f} seconds\n')

print(f"Results saved to {clip_scores_file} and {summary_file}")