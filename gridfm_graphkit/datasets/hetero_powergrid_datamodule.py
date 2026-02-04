import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import ConcatDataset
import torch.distributed as dist
from gridfm_graphkit.io.param_handler import (
    NestedNamespace,
    load_normalizer,
    get_task_transforms,
)
from gridfm_graphkit.datasets.powergrid_hetero_dataset import HeteroGridDatasetDisk
import os
import os.path as osp
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger


class LitGridHeteroDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for power grid datasets using task-based structure.

    This datamodule handles loading and preprocessing of power grid graph datasets
    from the task-based directory structure (data_tasks/{task}/train, valid, test).
    Splits are already defined in the folder structure, so no shuffling or splitting is performed.

    Args:
        args (NestedNamespace): Experiment configuration.
        data_dir (str, optional): Root directory for datasets. Defaults to "./data".

    Attributes:
        batch_size (int): Batch size for all dataloaders. From ``args.training.batch_size``
        data_normalizers (list): List of data normalizers.
        train_datasets (list): Train datasets.
        val_datasets (list): Validation datasets.
        test_datasets (list): Test datasets (kept separate, not concatenated).
        train_dataset_multi (ConcatDataset): Concatenated train datasets.
        val_dataset_multi (ConcatDataset): Concatenated validation datasets.
        _is_setup_done (bool): Tracks whether `setup` has been executed to avoid repeated processing.

    Methods:
        setup(stage):
            Load and preprocess datasets from data_tasks/{task}/train, valid, test folders.
            Normalizer is fitted only on training data; validation and test use those stats.
            Handles distributed preprocessing safely.
        train_dataloader():
            Returns a DataLoader for concatenated training datasets (no shuffling).
        val_dataloader():
            Returns a DataLoader for concatenated validation datasets.
        test_dataloader():
            Returns a list of DataLoaders, one per test dataset.
        predict_dataloader():
            Returns a list of DataLoaders, one per test dataset for prediction.

    Notes:
        - Preprocessing is only performed on rank 0 in distributed settings.
        - No shuffling or splitting is performed - splits are defined by folder structure.
        - Normalizer is fitted on training data only; same stats used for validation and test.
        - Test datasets are kept separate (not concatenated) - each gets its own DataLoader.

    Example:
        ```python
        from gridfm_graphkit.datasets.hetero_powergrid_datamodule import LitGridHeteroDataModule
        from gridfm_graphkit.io.param_handler import NestedNamespace
        import yaml

        with open("config/config.yaml") as f:
            base_config = yaml.safe_load(f)
        args = NestedNamespace(**base_config)

        datamodule = LitGridHeteroDataModule(args, data_dir="./data")

        datamodule.setup("fit")
        train_loader = datamodule.train_dataloader()
        ```
    """

    def __init__(self, args: NestedNamespace, data_dir: str = "./data"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = int(args.training.batch_size)
        self.args = args
        self.data_normalizers = []
        self.train_datasets = []
        self.val_datasets = []
        self.test_datasets = []
        self.test_dataset_names = []  # Store dataset names for test datasets
        self._is_setup_done = False

    def setup(self, stage: str):
        if self._is_setup_done:
            print(f"Setup already done for stage={stage}, skipping...")
            return
        

        
        # First, process train split to fit normalizer
        train_split_dir = os.path.join(self.data_dir, "train")
        train_split_root = train_split_dir  # Use split dir as root
        
        data_normalizer_train = load_normalizer(args=self.args)
        self.data_normalizers.append(data_normalizer_train)
        
        if os.path.exists(train_split_dir):
            # Run preprocessing only on rank 0
            if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
                print(f"Pre-processing train dataset on rank 0")
                _ = HeteroGridDatasetDisk(  # just to trigger processing
                    root=train_split_root,
                    norm_method=self.args.data.normalization,
                    data_normalizer=data_normalizer_train,
                    transform=get_task_transforms(args=self.args),
                )
            
            # All ranks wait here until processing is done
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            
            train_dataset = HeteroGridDatasetDisk(
                root=train_split_root,
                norm_method=self.args.data.normalization,
                data_normalizer=data_normalizer_train,
                transform=get_task_transforms(args=self.args),
            )
            self.train_datasets.append(train_dataset)
        
        # Process valid and test splits (they will use the normalizer stats from train)
        # Load train stats path to pass to other splits
        train_stats_path = os.path.join(train_split_root, "processed", f"data_stats_{self.args.data.normalization}.pt")
        
        # Process valid split (single dataset combining all cases)
        valid_split_dir = os.path.join(self.data_dir, "valid")
        if os.path.exists(valid_split_dir):
            valid_split_root = valid_split_dir
            
            data_normalizer_valid = load_normalizer(args=self.args)
            self.data_normalizers.append(data_normalizer_valid)

            # Run preprocessing only on rank 0
            if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
                print(f"Pre-processing valid dataset on rank 0")
                _ = HeteroGridDatasetDisk(  # just to trigger processing
                    root=valid_split_root,
                    norm_method=self.args.data.normalization,
                    data_normalizer=data_normalizer_valid,
                    transform=get_task_transforms(args=self.args),
                )
            
            # All ranks wait here until processing is done
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            
            valid_dataset = HeteroGridDatasetDisk(
                root=valid_split_root,
                norm_method=self.args.data.normalization,
                data_normalizer=data_normalizer_valid,
                transform=get_task_transforms(args=self.args),
            )
            self.val_datasets.append(valid_dataset)
        
        # Process test split - create separate dataset for each case_folder/grid_type/subdir combination
        test_split_dir = os.path.join(self.data_dir, "test")
        if os.path.exists(test_split_dir):
            # Find all case folders (case14, case30, etc.)
            case_folders = [
                d for d in os.listdir(test_split_dir)
                if os.path.isdir(os.path.join(test_split_dir, d)) and d.startswith("case")
            ]
            
            for case_folder in case_folders:
                case_dir = os.path.join(test_split_dir, case_folder)
                
                # Find all grid_type folders (n, n-1, n-2, etc.)
                grid_types = [
                    d for d in os.listdir(case_dir)
                    if os.path.isdir(os.path.join(case_dir, d)) and not d.startswith('.') and d != 'processed'
                ]
                
                for grid_type in grid_types:
                    grid_type_dir = os.path.join(case_dir, grid_type)
                    
                    # Find all subdirectories (feasible, nose, around_nose, etc.)
                    subdirs = [
                        d for d in os.listdir(grid_type_dir)
                        if os.path.isdir(os.path.join(grid_type_dir, d)) and not d.startswith('.')
                    ]
                    
                    for subdir in subdirs:
                        dataset_dir = os.path.join(grid_type_dir, subdir)
                        dataset_root = dataset_dir  # Use subdir as root (contains raw/ folder)
                        
                        data_normalizer_test = load_normalizer(args=self.args)
                        self.data_normalizers.append(data_normalizer_test)
                        
                        # Run preprocessing only on rank 0
                        if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
                            print(f"Pre-processing test dataset for {case_folder}/{grid_type}/{subdir} on rank 0")
                            _ = HeteroGridDatasetDisk(  # just to trigger processing
                                root=dataset_root,
                                norm_method=self.args.data.normalization,
                                data_normalizer=data_normalizer_test,
                                transform=get_task_transforms(args=self.args),
                            )
                        
                        # All ranks wait here until processing is done
                        if torch.distributed.is_available() and torch.distributed.is_initialized():
                            torch.distributed.barrier()
                        
                        dataset_name = f"test_{case_folder}_{grid_type}_{subdir}"
                        test_dataset = HeteroGridDatasetDisk(
                            root=dataset_root,
                            norm_method=self.args.data.normalization,
                            data_normalizer=data_normalizer_test,
                            transform=get_task_transforms(args=self.args),
                        )
                        self.test_datasets.append(test_dataset)
                        self.test_dataset_names.append(dataset_name)
            
        self.train_dataset_multi = ConcatDataset(self.train_datasets) if self.train_datasets else None
        self.val_dataset_multi = ConcatDataset(self.val_datasets) if self.val_datasets else None
        # Note: test_datasets are kept separate (not concatenated) - each gets its own DataLoader
        
        self._is_setup_done = True
        
        print("Length of train datasets: ", len(self.train_dataset_multi))
        print("Length of valid datasets: ", len(self.val_dataset_multi))
        print("Combined length of train and valid datasets: ", len(self.train_dataset_multi) + len(self.val_dataset_multi))
        print("Number of test datasets: ", len(self.test_datasets))
        for i, dataset in enumerate(self.test_datasets):
            # name:
            print(f"Name: {self.test_dataset_names[i]}")
            print(f"Length: {len(dataset)}")

        # Log the same dataset summary information to MLflow, if an MLFlowLogger is active.
        logger = getattr(getattr(self, "trainer", None), "logger", None)
        if isinstance(logger, MLFlowLogger):
            run_id = logger.run_id
            experiment = logger.experiment
            try:
                if self.train_dataset_multi is not None:
                    experiment.log_param(
                        run_id,
                        "dataset_train_len",
                        len(self.train_dataset_multi),
                    )
                if self.val_dataset_multi is not None:
                    experiment.log_param(
                        run_id,
                        "dataset_valid_len",
                        len(self.val_dataset_multi),
                    )
                    experiment.log_param(
                        run_id,
                        "dataset_train_plus_valid_len",
                        len(self.train_dataset_multi) + len(self.val_dataset_multi),
                    )
                experiment.log_param(
                    run_id,
                    "dataset_test_count",
                    len(self.test_datasets),
                )
                for i, dataset in enumerate(self.test_datasets):
                    name = self.test_dataset_names[i]
                    length = len(dataset)
                    experiment.log_param(run_id, f"test_dataset_{i}_name", name)
                    experiment.log_param(run_id, f"test_dataset_{i}_len", length)
            except Exception as e:
                print(f"Warning: failed to log dataset summary to MLflow: {e}")

    def train_dataloader(self):
        if self.train_dataset_multi is None:
            raise ValueError("No training datasets found. Make sure train split exists.")
        return DataLoader(
            self.train_dataset_multi,
            batch_size=self.batch_size,
            shuffle=True, 
            num_workers=self.args.data.workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        if self.val_dataset_multi is None:
            raise ValueError("No validation datasets found. Make sure valid split exists.")
        return DataLoader(
            self.val_dataset_multi,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.data.workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return [
            DataLoader(
                i,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.args.data.workers,
                pin_memory=True,
            )
            for i in self.test_datasets
        ]

    def predict_dataloader(self):
        return [
            DataLoader(
                i,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.args.data.workers,
                pin_memory=True,
            )
            for i in self.test_datasets
        ]
