"""
This script contains all the dataset classes required for the model.
"""

import torch
import numpy as np
import xarray as xr
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Dict, Tuple, Union


class ARTrainingDataset(Dataset):
    """
    Dataset class for training the DeepAR model.
    Loads input .nc files (IVT, IVT_u, IVT_v) and corresponding ground truth .nc files.
    
    Input files: Yearly files with data for a subset of days at 6-hourly intervals
    GT files: Yearly files with data for all days at 6-hourly intervals
    
    The dataset matches timestamps between input and GT files to create aligned pairs.
    Handles normalization of IVT data. SAM will handle padding to 1024x1024.
    """
    
    def __init__(
        self,
        input_files: List[Union[str, Path]],
        gt_files: List[Union[str, Path]],
        ivt_vars: Tuple[str, str, str] = ("ivt", "ivtu", "ivtv"),
        gt_var: str = "ar_mask",
        time_dim: str = "time",
        ivt_mean: Tuple[float, float, float] = (250.0, 50.0, 50.0),
        ivt_std: Tuple[float, float, float] = (100.0, 75.0, 75.0),
        use_memory_mapping: bool = True,
        chunk_size = 'auto'
    ):
        """
        Args:
            input_files: List of paths to yearly input .nc files (IVT_YYYY.nc format)
            gt_files: List of paths to yearly GT .nc files (PIKARTV1_*_YYYY.nc format)
            ivt_vars: Tuple of variable names for (IVT, IVT_u, IVT_v) in .nc files
            gt_var: Variable name for ground truth mask in .nc files
            time_dim: Name of time dimension in .nc files
            ivt_mean: Mean values for (IVT, IVT_u, IVT_v)
            ivt_std: Std values for (IVT, IVT_u, IVT_v)
            use_memory_mapping: Whether to use memory mapping for large datasets
            chunk_size: Chunk size for memory mapping (if used)
        """
        self.input_files = [Path(f) for f in input_files]
        self.gt_files = [Path(f) for f in gt_files]
        self.ivt_vars = ivt_vars
        self.gt_var = gt_var
        self.time_dim = time_dim
        self.use_memory_mapping = use_memory_mapping
        self.chunk_size = chunk_size
        
        # Store normalization parameters
        self.ivt_mean = np.array(ivt_mean, dtype=np.float32).reshape(3, 1, 1)
        self.ivt_std = np.array(ivt_std, dtype=np.float32).reshape(3, 1, 1)
        
        if len(self.input_files) != len(self.gt_files):
            raise ValueError(f"Number of input files ({len(self.input_files)}) must match number of GT files ({len(self.gt_files)})")
        
        if self.use_memory_mapping:
            print("Opening dataset files with memory mapping...")
            try:
                self.input_datasets = [xr.open_dataset(f, chunks=self.chunk_size) for f in self.input_files]
                self.gt_datasets = [xr.open_dataset(f) for f in self.gt_files]
            except Exception as e:
                print(f"Error opening dataset files: {e}")
                # Clean up any partially opened datasets
                if hasattr(self, 'input_datasets') and self.input_datasets:
                    for ds in self.input_datasets:
                        try:
                            ds.close()
                        except:
                            pass
                raise
        else:
            self.input_datasets = None
            self.gt_datasets = None

        print("Building dataset index...")
        self._build_index()

    def _build_index(self):
        """
        Build index by matching timestamps between input and GT files.
        Each index entry stores: (file_idx, input_time_idx, gt_time_idx).

        Since input files contain subset of days (1, 6, 11, 16, 21, 26, 31),
        we need to find matching timestamps in the GT files (which have all days).
        """
        self.index = []
        total_matched = 0
        total_unmatched = 0

        for file_idx in range(len(self.input_files)):
            if self.use_memory_mapping:
                input_ds = self.input_datasets[file_idx]
                gt_ds = self.gt_datasets[file_idx]
            else:
                input_ds = xr.open_dataset(self.input_files[file_idx])
                gt_ds = xr.open_dataset(self.gt_files[file_idx])
            input_times = input_ds[self.time_dim].values
            gt_times = gt_ds[self.time_dim].values
            # Match each input timestamp to GT timestamp
            for input_time_idx, input_time in enumerate(input_times):
                matching_indices = np.where(gt_times == input_time)[0]

                if len(matching_indices) > 0:
                    gt_time_idx = int(matching_indices[0])
                    self.index.append((file_idx, input_time_idx, gt_time_idx))
                    total_matched += 1
                else:
                    total_unmatched += 1
                    print(f"Warning: No matching GT timestamp for input time {input_time} in file {self.input_files[file_idx]}")

            if not self.use_memory_mapping:
                input_ds.close()
                gt_ds.close()

        print(f"Total matched samples: {total_matched}")
        print(f"Total unmatched samples: {total_unmatched}")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Tuple[int, int], Dict[str, str]]]:
        """
        Load a single sample (input + GT).

        Args:
            idx: Index of the sample to load

        Returns a dictionary containing:
            - 'image': Normalized IVT tensor (3, H, W) in original size
            - 'original_size': Original spatial dimensions (H, W)
            - 'gt_mask': Ground truth mask (1, H, W)
        
        Note: SAM will handle padding to 1024x1024 during forward pass.
        """
        file_idx, input_time_idx, gt_time_idx = self.index[idx]
        
        # Load input data
        if self.use_memory_mapping:
            input_ds = self.input_datasets[file_idx]
        else:
            input_ds = xr.open_dataset(self.input_files[file_idx])

        ivt = input_ds[self.ivt_vars[0]].isel({self.time_dim: input_time_idx}).values
        ivt_u = input_ds[self.ivt_vars[1]].isel({self.time_dim: input_time_idx}).values
        ivt_v = input_ds[self.ivt_vars[2]].isel({self.time_dim: input_time_idx}).values

        input_array = np.stack([ivt, ivt_u, ivt_v], axis=0)
        original_size = input_array.shape[1:]  # (H, W)

        # Load ground truth mask
        if self.use_memory_mapping:
            gt_ds = self.gt_datasets[file_idx]
        else:
            gt_ds = xr.open_dataset(self.gt_files[file_idx])

        gt_mask = gt_ds[self.gt_var].isel({self.time_dim: gt_time_idx}).values

        if not self.use_memory_mapping:
            input_ds.close()
            gt_ds.close()
        
        # Normalise input data:
        input_array = (input_array - self.ivt_mean) / self.ivt_std

        # Convert to tensors
        input_tensor = torch.from_numpy(input_array).float()  # (3, H, W)
        gt_mask_tensor = torch.from_numpy(gt_mask).float().unsqueeze(0) # (1, H, W)

        return {
            'image': input_tensor,
            'original_size': original_size,
            'gt_mask': gt_mask_tensor,
        }
    
    def __del__(self):
        """Closes all open datasets when the object is destroyed."""
        if self.use_memory_mapping and hasattr(self, 'input_datasets'):
            if self.input_datasets is not None:
                for ds in self.input_datasets:
                    try:
                        ds.close()
                    except Exception:
                        pass  # Ignore errors during cleanup
            if hasattr(self, 'gt_datasets') and self.gt_datasets is not None:
                for ds in self.gt_datasets:
                    try:
                        ds.close()
                    except Exception:
                        pass     




class ARInferenceDataset(Dataset):
    """
    Dataset class for inference with the DeepAR model.
    Loads only input .nc files (IVT, IVT_u, IVT_v) without ground truth.
    
    Handles normalization of IVT data. SAM will handle padding to 1024x1024.
    """
    
    def __init__(
        self,
        input_files: List[Union[str, Path]],
        ivt_vars: Tuple[str, str, str] = ("ivt", "ivtu", "ivtv"),
        time_dim: str = "time",
        ivt_mean: Tuple[float, float, float] = (250.0, 50.0, 50.0),  
        ivt_std: Tuple[float, float, float] = (100.0, 75.0, 75.0),
        use_memory_mapping: bool = True,
        chunk_size = 'auto',
        return_metadata: bool = True,
    ):
        """
        Args:
            input_files: List of paths to input .nc files containing IVT data
            ivt_vars: Tuple of variable names for (IVT, IVT_u, IVT_v) in .nc files
            time_dim: Name of time dimension in .nc files
            ivt_mean: Mean values for (IVT, IVT_u, IVT_v)
            ivt_std: Std values for (IVT, IVT_u, IVT_v)
            use_memory_mapping: Whether to use memory mapping for large datasets
            chunk_size: Chunk size for memory mapping (if used)
            return_metadata: Whether to return metadata (timestamp, coordinates, etc.)
        """
        self.input_files = [Path(f) for f in input_files]
        self.ivt_vars = ivt_vars
        self.time_dim = time_dim
        self.use_memory_mapping = use_memory_mapping
        self.chunk_size = chunk_size
        self.return_metadata = return_metadata
        
        # Store normalization parameters
        self.ivt_mean = np.array(ivt_mean, dtype=np.float32).reshape(3, 1, 1)
        self.ivt_std = np.array(ivt_std, dtype=np.float32).reshape(3, 1, 1)

        if self.use_memory_mapping:
            print("Opening dataset files with memory mapping...")
            self.input_datasets = [xr.open_dataset(f, chunks=self.chunk_size) for f in self.input_files]
        else:
            self.input_datasets = None

        # Build index
        print("Building dataset index...")
        self._build_index()
        print(f"Dataset ready with {len(self.index)} samples.")

    def _build_index(self):
        """
        Build simple sequential index: (file_idx, time_idx).
        """
        self.index = []
        
        for file_idx, input_file in enumerate(self.input_files):
            if self.use_memory_mapping:
                ds = self.input_datasets[file_idx]
            else:
                ds = xr.open_dataset(input_file)
            
            n_timesteps = len(ds[self.time_dim])

            for time_idx in range(n_timesteps):
                self.index.append((file_idx, time_idx))
                
            if not self.use_memory_mapping:
                ds.close()
    
    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, Dict]]:
        """
        Load a single input sample.
        
        Returns a dictionary containing:
            - 'image': Normalized input tensor (3, H, W) in original size
            - 'original_size': Original spatial dimensions (H, W)
            - 'metadata': (Optional) Dict with timestamp and file info
        """
        file_idx, time_idx = self.index[idx]

        if self.use_memory_mapping:
            input_ds = self.input_datasets[file_idx]
        else:
            input_ds = xr.open_dataset(self.input_files[file_idx])

        # Extract IVT variables
        ivt = input_ds[self.ivt_vars[0]].isel({self.time_dim: time_idx}).values
        ivt_u = input_ds[self.ivt_vars[1]].isel({self.time_dim: time_idx}).values
        ivt_v = input_ds[self.ivt_vars[2]].isel({self.time_dim: time_idx}).values

        # Stack and get metadata
        input_array = np.stack([ivt, ivt_u, ivt_v], axis=0)
        original_size = input_array.shape[1:]  # (H, W)

        if self.return_metadata:
            timestamp = input_ds[self.time_dim].isel({self.time_dim: time_idx}).values
            metadata = {
                'file': str(self.input_files[file_idx]),
                'timestamp': str(timestamp),
                'time_idx': time_idx
            }
        
        if not self.use_memory_mapping:
            input_ds.close()
        
        # Normalize
        input_array = (input_array - self.ivt_mean) / self.ivt_std
        # Convert to tensor
        input_tensor = torch.from_numpy(input_array).float()  # (3, H, W)

        result = {
            'image': input_tensor,
            'original_size': original_size,
        }

        if self.return_metadata:
            result['metadata'] = metadata

        return result

    def __del__(self):
        """Closes all open datasets."""
        if self.use_memory_mapping and hasattr(self, 'input_datasets'):
            if self.input_datasets is not None:
                for ds in self.input_datasets:
                    try:
                        ds.close()
                    except Exception:
                        pass