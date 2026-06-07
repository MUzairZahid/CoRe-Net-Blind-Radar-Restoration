import argparse
import os
import torch
from utils import *
from models import initialize_models
from dataset import *
from train import *
from evaluation import *

# Function to append to the folder name only if the argument is provided
def append_if_provided(value, label, out_folder_parts):
    if value is not None and value != "":
        out_folder_parts.append(f"{label}{value}")

# python main.py --device cuda:0 --Eps 1.0 --Beta 10.0 --Phi 1.0 --psnr_pred_loss mse --scheduler none

def main():
    # Setup command-line arguments for configuring the training and evaluation process
    parser = argparse.ArgumentParser(description='Train and/or Evaluate a CycleGAN model on a dataset')
    parser.add_argument('--master_type', type=str, choices=['standard', 'distortionAware'], default='distortionAware', help='Choose the type of master model: standard or distortion-aware')
    parser.add_argument('--Q', type=int, default=3, help='Set q value for SelfONN layers, with 1 representing a conventional CNN.')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'both'], default='both', help='Define operation mode: train, evaluate, or both.')
    parser.add_argument('--epochs', type=int, default=1000, help='Specify the number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size.')
    parser.add_argument('--device', type=str, default='cuda', help='Select the computation device: cpu or cuda.')
    parser.add_argument('--out_folder', type=str, default='', help='Output folder for saving model weights and logs.')
    parser.add_argument('--data_folder', type=str, default='Prepared_Dataset', help='Folder containing the dataset.')
    parser.add_argument('--dataset', type=str, choices=['base', 'extended'], default='extended', help='Choose the dataset: base or extended.')
    parser.add_argument('--normalize', type=bool, default=True, help='Enable data normalization.')
    parser.add_argument('--val_interval', type=int, default=1, help='Interval for logging and evaluation.')
    parser.add_argument('--targetpsnr', type=int, default=40, help='Target PSNR value.')
    parser.add_argument('--Eps', type=float, default=1.0, help='Weight for the loss component Eps.')
    parser.add_argument('--Beta', type=float, default=10.0, help='Weight for the loss component Beta.')
    parser.add_argument('--Phi', type=float, default=1.0, help='Weight for the loss component Phi.')
    parser.add_argument('--dataset_fraction', type=float, default=1, help='Fraction of dataset. 1 corresponds to all data.')
    parser.add_argument('--rec_loss', type=str, choices=['psnr', 'l1', 'mse'], default='psnr', help='Loss for reconstruction by master.')
    parser.add_argument('--psnr_pred_loss', type=str, choices=['l1', 'mse'], default='mse', help='Loss for psnr prediction by master.')
    parser.add_argument('--scheduler', type=str, choices=['none', 'cosine', 'cosineWarm'], default='cosine', help='Learning rate scheduler to use: none or cosine')
    parser.add_argument('--lr_A', type=float, default=0.005, help='Apprentice Learning rate.')
    parser.add_argument('--lr_M', type=float, default=0.005, help='Master Learning rate.')
    parser.add_argument('--T_max', type=int, default=100, help='T max for scheduler.')
    parser.add_argument('--continue_training', action='store_true', help='Continue training from saved weights if available')


    args = parser.parse_args()

    # Initialize the out_folder
    out_folder_parts = [args.out_folder]

    # Add components to out_folder based on whether they are provided
    append_if_provided(args.dataset, "", out_folder_parts)
    append_if_provided(args.targetpsnr, "T", out_folder_parts)
    append_if_provided(args.Eps, "Eps", out_folder_parts)
    append_if_provided(args.Beta, "Beta", out_folder_parts)
    append_if_provided(args.Phi, "Phi", out_folder_parts)
    append_if_provided(args.master_type, "", out_folder_parts)
    append_if_provided(args.rec_loss, "", out_folder_parts)
    append_if_provided(args.psnr_pred_loss, "", out_folder_parts)
    append_if_provided(args.scheduler, "", out_folder_parts)

    # Join the parts with underscores and set it as the output folder name
    args.out_folder = '_'.join(out_folder_parts)


    print(f"Configured output folder: {args.out_folder}")
    os.makedirs(args.out_folder, exist_ok=True)  # Ensure the output directory exists

    config_path = os.path.join(args.out_folder, 'config.json')
    save_config(args, config_path)
    print(f"Saved configuration to {config_path}")

    # Construct the path for the dataset based on the user's choices.
    dataset_filename = f"{args.dataset}_dataset_k0.pickle"
    args.dataset_path = os.path.join(args.data_folder, dataset_filename)
    print(f"Selected dataset filename: {dataset_filename}")

    # Initialize models based on the provided configuration
    A, M = initialize_models(args)

    args.model_path = os.path.join(args.out_folder, 'model_max_valSNR.pth')
    config_path = os.path.join(args.out_folder, 'config.json')

    # Load and prepare dataset based on the selected dataset
    dataloaders = load_and_prepare_dataset(args, num_workers=4, pin_memory=True)

    # Train or evaluate the model based on the specified mode
    if args.mode in ['train', 'both']:
        if args.continue_training:
            checkpoint = torch.load(args.model_path, map_location=args.device)
            A.load_state_dict(checkpoint['A_state_dict'])
            M.load_state_dict(checkpoint['M_state_dict'])
            print('Continue training with saved weights. ')
        # Train the model
        train_coreNet(dataloaders, A, M, args)

    if args.mode in ['evaluate', 'both']:
        checkpoint = torch.load(args.model_path, map_location=args.device)
        A.load_state_dict(checkpoint['A_state_dict'])

        # Write results to a text file in the specified output folder
        results_file_path = os.path.join(args.out_folder, 'evaluation_results_22.txt')

        # Evaluate the model on the specified dataset
        if args.dataset == "base":
            results = evaluate_model_on_test_data_awgn(A, dataloaders, args.device)
            with open(results_file_path, 'w') as file:
                for snr_level, values in results.items():
                    mean_snr = np.mean(values)
                    file.write(f'True SNR {snr_level} dB: Mean Restored SNR: {mean_snr:.2f} dB\n')
        elif args.dataset == "extended":
            results = evaluate_model_on_test_data_blind(A, dataloaders, args.device)
            with open(results_file_path, 'w') as file:
                for key, value in results.items():
                    file.write(f'{key}: {value:.6f}\n')

        print(f'Results saved to {results_file_path}')

if __name__ == "__main__":
    main() 