import torch
import json
import argparse
import torch.nn as nn
import os


class PSNRLoss(nn.Module):
    def __init__(self, max_value=1.0):
        super(PSNRLoss, self).__init__()
        self.max_value = max_value

    def forward(self, signal, reference):
        mse_i = torch.mean((signal[:, 0, :] - reference[:, 0, :]) ** 2, dim=-1)
        mse_q = torch.mean((signal[:, 1, :] - reference[:, 1, :]) ** 2, dim=-1)
        mse = mse_i + mse_q
        psnr = 10 * torch.log10(self.max_value ** 2 / mse)
        
        # Return negative PSNR as loss (since we minimize the loss)
        return -psnr.mean()
    
def calculate_psnr(signal, reference, max_value=1.0):
    mse_i = torch.mean((signal[:, 0, :] - reference[:, 0, :])**2, dim=-1)
    mse_q = torch.mean((signal[:, 1, :] - reference[:, 1, :])**2, dim=-1)
    mse = mse_i + mse_q
    psnr = 10 * torch.log10(max_value ** 2 / mse)
    return psnr

def calculate_snr(noisy_signal, clean_signal):
    """
    Calculate the Signal-to-Noise Ratio (SNR) for batched inputs by using
    the absolute squares of the I and Q components.

    Parameters:
    noisy_signal (Tensor): The noisy signals, assumed to be of shape (batch_size, 2, signal_length),
                           where the second dimension contains I and Q components.
    clean_signal (Tensor): The clean signals, assumed to be of shape (batch_size, 2, signal_length),
                           where the second dimension contains I and Q components.

    Returns:
    Tensor: The SNR values in dB for each sample in the batch.
    """
    # Calculate the noise
    noise = noisy_signal - clean_signal

    # Calculate signal and noise power using their I and Q components
    signal_power_i = torch.mean(clean_signal[:, 0, :]**2, dim=-1)
    signal_power_q = torch.mean(clean_signal[:, 1, :]**2, dim=-1)
    noise_power_i = torch.mean(noise[:, 0, :]**2, dim=-1)
    noise_power_q = torch.mean(noise[:, 1, :]**2, dim=-1)

    # Total signal and noise power
    signal_power = signal_power_i + signal_power_q
    noise_power = noise_power_i + noise_power_q

    # Calculate SNR, adding a small epsilon to noise_power to avoid division by zero
    snr = signal_power / (noise_power + 1e-6)
    snr_db = 10 * torch.log10(snr)

    return snr_db

def calculate_spectrogram_torch(signals, n_fft=256, hop_length=128, win_length=256, power=2.0):
    """
    Calculate the spectrogram of signals using PyTorch, keeping the computation on the GPU.
    
    Parameters:
    - signals: Tensor of shape (batch_size, 2, signal_length)
    - n_fft: Number of FFT points
    - hop_length: Number of samples between successive frames
    - win_length: Each frame of audio is windowed by `window()` and will have length `win_length`
    - power: Exponent for the magnitude spectrogram
    
    Returns:
    - Tensor of spectrograms of shape (batch_size, freq_bins, time_steps)
    """
    # signals is expected to be real-valued, with dimension (batch_size, 2, signal_length)
    signals = signals.permute(0, 2, 1)  # Now shape is (batch_size, signal_length, 2)
    # Make sure signals tensor is contiguous in memory, especially after reshaping
    signals = signals.contiguous()
    signals_complex = torch.view_as_complex(signals)  # Convert to complex tensor

    # Create Hanning window
    window = torch.hann_window(win_length).to(signals.device)
    
    # Compute STFT
    stft = torch.stft(signals_complex, n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True, window=window)
    
    # Compute spectrogram as the squared magnitude of the STFT
    spectrogram = torch.abs(stft)**power
    
    # Normalize spectrogram
    spectrogram_min = spectrogram.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    spectrogram_max = spectrogram.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    normalized_spectrogram = 2 * (spectrogram - spectrogram_min) / (spectrogram_max - spectrogram_min) - 1
    
    return normalized_spectrogram

def denormalize_signal(normalized_signal, original_min, original_max):
    """
    Denormalize the signal from the range [-1, 1] back to its original range.

    Parameters:
    normalized_signal (numpy array or torch tensor): The normalized signal.
    original_min (float): The minimum value of the original signal before normalization.
    original_max (float): The maximum value of the original signal before normalization.

    Returns:
    numpy array or torch tensor: Denormalized signal.
    """
    return ((normalized_signal + 1) / 2) * (original_max - original_min) + original_min


def save_config(args, config_path):
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return argparse.Namespace(**config)

def save_best_model(A, M, model_path):
    torch.save({
        'A_state_dict': A.state_dict(),
        'M_state_dict': M.state_dict(),
    }, model_path)
    print(f"Saved best model with improved validation SNR to {model_path}")


def write_losses_to_tensorboard(
    loss_A_b,
    loss_A_restoration_fidelity_b,
    loss_A_psnr_time_b,
    loss_A_psnr_freq_b,
    loss_M_b,
    loss_M_real_b,
    loss_M_fake_b,
    mean_out_M_real_b,
    mean_real_labels_b,
    mean_out_M_fake_b,
    mean_fake_labels_b,
    writer,
    epoch,
    phase,
    lr=None
):
    """
    Logs epoch-level averaged losses/metrics to TensorBoard.

    IMPORTANT:
    In your current run_epoch(), these values are already averaged over samples.
    Therefore, do NOT divide again by len(dataloaders[phase]).
    """

    # Apprentice
    writer.add_scalar(f"{phase}_Loss_Apprentice", loss_A_b, epoch)
    writer.add_scalar(f"{phase}_Loss_Apprentice_fidelity", loss_A_restoration_fidelity_b, epoch)
    writer.add_scalar(f"{phase}_Loss_Apprentice_time", loss_A_psnr_time_b, epoch)
    writer.add_scalar(f"{phase}_Loss_Apprentice_freq", loss_A_psnr_freq_b, epoch)

    # Master
    writer.add_scalar(f"{phase}_Loss_Master", loss_M_b, epoch)
    writer.add_scalar(f"{phase}_Loss_Master_real", loss_M_real_b, epoch)
    writer.add_scalar(f"{phase}_Loss_Master_fake", loss_M_fake_b, epoch)

    # MR predicted vs actual PSNR
    writer.add_scalar(f"{phase}_psnr_predicted_real", mean_out_M_real_b, epoch)
    writer.add_scalar(f"{phase}_psnr_actual_real", mean_real_labels_b, epoch)

    writer.add_scalar(f"{phase}_psnr_predicted_fake", mean_out_M_fake_b, epoch)
    writer.add_scalar(f"{phase}_psnr_actual_fake", mean_fake_labels_b, epoch)


    # Learning rate
    if phase == "train" and lr is not None:
        writer.add_scalar("Learning_rate", lr, epoch)

    # Force writing to disk, useful on cluster jobs
    writer.flush()