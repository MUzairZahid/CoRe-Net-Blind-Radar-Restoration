import torch
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import pickle

class RadarSignalDataset(Dataset):
    def __init__(self, clean_signals, distorted_signals, labels, snr_distorted, distortions = None, normalize=True):
        """
        Custom Dataset for radar signals with optional normalization.
        """
        # Convert numpy arrays to torch tensors once rather than converting them every time __getitem__ is called
        self.clean_signals = torch.tensor(clean_signals, dtype=torch.float32)
        self.distorted_signals = torch.tensor(distorted_signals, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long).squeeze()
        self.labels = self.labels - 1  # Convert from 1-indexed to 0-indexed
        self.snr_distorted = torch.tensor(snr_distorted, dtype=torch.float32) 
        self.distortions = torch.tensor(distortions, dtype=torch.float32) if distortions is not None else None
        self.normalize = normalize

        # Store min and max values for each channel of each signal
        self.clean_min_max = self.calculate_min_max(self.clean_signals)
        self.distorted_min_max = self.calculate_min_max(self.distorted_signals)
                

    def __len__(self):
        return len(self.clean_signals)

    def __getitem__(self, idx):
        clean_signal = self.clean_signals[idx]
        distorted_signal = self.distorted_signals[idx]
        label = self.labels[idx]
        snr_distorted = self.snr_distorted[idx]
        distortions = self.distortions[idx] if self.distortions is not None else torch.tensor([])

        clean_min, clean_max = self.clean_min_max[idx,:,0].view(-1,1), self.clean_min_max[idx,:,1].view(-1,1) # torch.Size([2, 1])
        distorted_min, distorted_max = self.distorted_min_max[idx,:,0].view(-1,1), self.distorted_min_max[idx,:,1].view(-1,1)
        
        if self.normalize:
            clean_signal = self.normalize_signal(clean_signal, clean_min, clean_max)
            distorted_signal = self.normalize_signal(distorted_signal, distorted_min, distorted_max)

        return (clean_signal, clean_min, clean_max), \
               (distorted_signal, distorted_min, distorted_max), \
               label, snr_distorted, distortions

    @staticmethod
    def normalize_signal(signal, signal_min, signal_max):
        """
        Normalize the multi-channel signal to the range [-1, 1].

        Parameters:
        signal (numpy array): The signal to be normalized.
        signal_min (numpy array): Min values for each channel.
        signal_max (numpy array): Max values for each channel.

        Returns:
        numpy array: Normalized signal.
        """
        epsilon = 1e-8  # Added to prevent division by zero
        return 2 * ((signal - signal_min) / (signal_max - signal_min + epsilon)) - 1
    
    @staticmethod
    def calculate_min_max(signals):
        # Calculate the min and max across the last dimension (1024 samples)
        # and maintain the original dimensions of [49920, 2, 1] for compatibility
        min_vals = signals.min(dim=2, keepdim=True)[0]
        max_vals = signals.max(dim=2, keepdim=True)[0]

        # Combine the min and max into a single tensor for each signal
        # Resulting shape will be [49920, 2, 2] where the last dimension holds min and max values
        min_max_vals = torch.cat((min_vals, max_vals), dim=2)
    
        return min_max_vals


def load_and_prepare_dataset(args, num_workers=1, pin_memory=False):
    """
    Load the dataset and prepare it for model training, with options for pinned memory and multiple worker processes.
    Also supports using a subset of the dataset.

    Parameters:
    batch_size (int): Batch size for DataLoader.
    dataset_path (str): Path to the folder containing saved datasets.
    normalize (bool): Whether to normalize the signals.
    num_workers (int): Number of subprocesses to use for data loading.
    pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
    subset_fraction (float): Fraction of the dataset to use (between 0 and 1).

    Returns:
    dict: A dictionary containing DataLoader objects for training, validation, and test sets.
    """
    import gc
    
    # Load the dataset
    with open(args.dataset_path, 'rb') as file:
        dataset = pickle.load(file)

    dataloaders = {}
    for split in ['train', 'validation', 'test']:
        # Check if 'distortions' key is present
        distortions = dataset[split].get('distortions', None)

        radar_dataset = RadarSignalDataset(
            dataset[split]['clean'],
            dataset[split]['noisy'],
            dataset[split]['label'],
            dataset[split]['SNR'],
            distortions,
            normalize=args.normalize
        )

        # BUGFIX: Apply subset if dataset_fraction < 1.0
        if args.dataset_fraction < 1.0:
            subset_size = int(args.dataset_fraction * len(radar_dataset))
            # Create random indices and subsets
            indices = np.random.choice(len(radar_dataset), subset_size, replace=False)
            subset = Subset(radar_dataset, indices)
            # FIX: Use the subset instead of the full dataset
            dataloaders[split] = DataLoader(
                subset,  # Changed from radar_dataset to subset
                batch_size=args.batch_size, 
                shuffle=(split == 'train'),
                num_workers=num_workers, 
                pin_memory=pin_memory
            )
        else:
            # Use full dataset
            dataloaders[split] = DataLoader(
                radar_dataset, 
                batch_size=args.batch_size, 
                shuffle=(split == 'train'),
                num_workers=num_workers, 
                pin_memory=pin_memory
            )
    
    # OPTIMIZATION: Delete the pickle data to free memory
    # The data has been converted to torch tensors in the dataset objects
    # so we can safely delete the numpy arrays from the pickle file
    del dataset
    gc.collect()
    
    return dataloaders