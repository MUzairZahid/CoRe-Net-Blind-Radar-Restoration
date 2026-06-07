import torch
import numpy as np
import time
import torch.nn.functional as F
from utils import denormalize_signal, calculate_psnr, calculate_snr


def evaluate_model_on_test_data_blind(G, dataloaders, device):
    G.eval()
    snr_noisy_list = []
    snr_restored_list = []
    mse_noisy_list = []
    mse_restored_list = []
    psnr_noisy_list = []
    psnr_restored_list = []
    
    total_inference_time = 0  # To accumulate total inference time
    num_samples = 0  # To count the number of samples processed
    
    with torch.no_grad():
        for (clean_signal, clean_min, clean_max), \
               (distorted_signal, distorted_min, distorted_max), \
               label, snr_distorted, distortions in dataloaders['test']:
            clean_signal = clean_signal.to(device)
            distorted_signal = distorted_signal.to(device)
            clean_min, clean_max = clean_min.to(device), clean_max.to(device)
            distorted_min, distorted_max = distorted_min.to(device), distorted_max.to(device)
            
            # Start timing the inference
            start_time = time.time()
            restored_signal = G(distorted_signal)
            # Stop timing the inference
            inference_time = time.time() - start_time
            
            total_inference_time += inference_time
            num_samples += clean_signal.size(0)
            
            # Denormalize signals
            clean_signal = denormalize_signal(clean_signal, clean_min, clean_max)
            distorted_signal = denormalize_signal(distorted_signal, distorted_min, distorted_max)
            restored_signal = denormalize_signal(restored_signal, clean_min, clean_max)

            # Calculate metrics for noisy input
            snr_noisy = snr_distorted.cpu().numpy()
            mse_noisy = F.mse_loss(distorted_signal, clean_signal, reduction='none').mean(dim=[1, 2]).cpu().numpy()
            psnr_noisy = calculate_psnr(distorted_signal, clean_signal).cpu().numpy()

            # Calculate metrics for restored signal
            snr_restored = calculate_snr(restored_signal, clean_signal).cpu().numpy()
            mse_restored = F.mse_loss(restored_signal, clean_signal, reduction='none').mean(dim=[1, 2]).cpu().numpy()
            psnr_restored = calculate_psnr(restored_signal, clean_signal).cpu().numpy()
            
            snr_noisy_list.extend(snr_noisy)
            snr_restored_list.extend(snr_restored)
            mse_noisy_list.extend(mse_noisy)
            mse_restored_list.extend(mse_restored)
            psnr_noisy_list.extend(psnr_noisy)
            psnr_restored_list.extend(psnr_restored)
    
    avg_snr_noisy = np.mean(snr_noisy_list)
    avg_snr_restored = np.mean(snr_restored_list)
    avg_mse_noisy = np.mean(mse_noisy_list)
    avg_mse_restored = np.mean(mse_restored_list)
    avg_psnr_noisy = np.mean(psnr_noisy_list)
    avg_psnr_restored = np.mean(psnr_restored_list)
    
    # Calculate average inference time per sample
    avg_inference_time_per_sample = total_inference_time / num_samples
    
    print(f'Average Noisy SNR: {avg_snr_noisy:.2f} dB')
    print(f'Average Restored SNR: {avg_snr_restored:.2f} dB')
    print(f'Average Noisy MSE: {avg_mse_noisy:.6f}')
    print(f'Average Restored MSE: {avg_mse_restored:.6f}')
    print(f'Average Noisy PSNR: {avg_psnr_noisy:.2f} dB')
    print(f'Average Restored PSNR: {avg_psnr_restored:.2f} dB')
    print(f'Average Inference Time per Sample: {avg_inference_time_per_sample:.6f} seconds')
    
    return {
        'avg_snr_noisy': avg_snr_noisy,
        'avg_snr_restored': avg_snr_restored,
        'avg_mse_noisy': avg_mse_noisy,
        'avg_mse_restored': avg_mse_restored,
        'avg_psnr_noisy': avg_psnr_noisy,
        'avg_psnr_restored': avg_psnr_restored,
        'avg_inference_time_per_sample': avg_inference_time_per_sample
    }



def evaluate_model_on_test_data_awgn(G, dataloaders, device):
    """
    Evaluate the trained model on test data and calculate mean SNR for each specific SNR value.

    Parameters:
    - model_path (str): Path to the trained model file.
    - dataloaders (dict): Dataloaders for the dataset.
    - device (torch.device): Device to perform the evaluation on.
    """
    G.eval()
    # Assuming ResidualGenerator is your model class

    snr_results = {}  # Dictionary to store results
    all_snr = []

    # Create keys based on the range from -14 to 10 with step size of 2
    for snr in range(-14, 12, 2):  # Using range function with step size of 2
        snr_results[snr] = []  # Assigning None as placeholder value

    with torch.no_grad():
        for (clean_signal, clean_min, clean_max), \
            (distorted_signal, distorted_min, distorted_max), \
            label, snr_distorted_batch, distortions in dataloaders['test']:
            clean_signal, distorted_signal = clean_signal.to(device), distorted_signal.to(device)
            clean_min, clean_max = clean_min.to(device), clean_max.to(device)
            restored_signal = G(distorted_signal)

            clean_signal = denormalize_signal(clean_signal, clean_min, clean_max)
            restored_signal = denormalize_signal(restored_signal, clean_min, clean_max)

            snr_restored_batch = calculate_snr(restored_signal, clean_signal).cpu().numpy()
            for value, snr_value in zip(snr_restored_batch, snr_distorted_batch.cpu().numpy()):
                snr_results[snr_value[0]].append(value)
                all_snr.append(value)

    # Calculate and print the mean SNR for each SNR level
    for snr_level in sorted(snr_results.keys()):
        mean_snr = np.mean(snr_results[snr_level])
        print(f'True SNR {snr_level} dB: Mean Restored SNR: {mean_snr:.2f} dB')


    print(f'Mean SNR dB: {np.mean(all_snr):.2f} dB')

    return snr_results