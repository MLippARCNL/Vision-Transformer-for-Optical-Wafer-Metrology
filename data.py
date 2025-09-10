import sys
import torch
sys.path.append(r"C:\Users\dwolf\PycharmProjects\data_analysis_tools")

from pathlib import Path
from functools import partial
from h5py import File as H5File
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, RandomAffine, Normalize
from datamodule import H5DataModuleStatic, H5Dataset, DHMexpModuleStatic


class Data(H5DataModuleStatic):
    def __init__(self,
                 train_file: Path,
                 val_file: Path=None,
                 test_file: Path=None,
                 split_train: float=None,
                 augment: bool=True,
                 norm: bool=True,
                 budget: int=1024,
                 batch_size: int=int(512 // 8),
                 train_range: tuple=None,
                 ):
        """
        Args:
            train_file:
            val_file:
            test_file:
            split_train:
            augment:
            norm:
            budget:
            batch_size:
            train_range:
        """
        splits = {
            'W0_10': ['2025_02_12', '', '2', '2.2683959007263184', '4.348666667938232'],
            'W1_10': ['2025_02_13', '', '2', '2.6559672355651855', '4.532663345336914'],
            'W2_10': ['2025_02_12', '_3', '4', '2.2979037761688232', '4.215606689453125'],

            'W0_16': ['2025_02_17', '', '2', '2.7586565017700195', '4.773308753967285'],
            'W1_16': ['2025_03_25', '', '2', '2.984602928161621', '4.938925743103027'],
            'W2_16': ['2025_02_17', '_3', '4', '3.557494640350342', '5.704902172088623'],
        }

        _p = Path(r"\\sun\amitonova\group-folder\Users\Maximilian Lipp\Data")
        config = Path(f"{_p}/{splits[train_file][0]}/configuration.json")
        train = Path(f"{_p}/{splits[train_file][0]}/dataset{splits[train_file][1]}_cropped.h5")
        val = Path(f"{_p}/{splits[val_file][0]}/dataset_{splits[val_file][2]}_cropped.h5") if val_file is not None else None
        test = Path(f"{_p}/{splits[test_file][0]}/dataset_{splits[test_file][2]}_cropped.h5") if test_file is not None else None

        self.split_train = split_train
        case_1 = (self.split_train is not None and val is None and test is None)
        case_2 = (self.split_train is None and (val is not None and test is not None))
        assert (case_1 or case_2), \
            "Set split_train != None OR set test/val sets != None (not both at the same time)."

        self._mean = [float(splits[train_file][3]) for _ in range(3)] if self.split_train is not None else [
            float(splits[train_file][3]),
            float(splits[val_file][3]),
            float(splits[test_file][3])
        ]
        self._std = [float(splits[train_file][4]) for _ in range(3)] if self.split_train is not None else [
            float(splits[train_file][4]),
            float(splits[val_file][4]),
            float(splits[test_file][4])
        ]
        if not norm:
            self._mean, self._std = [0 for _ in range(3)], [1 for _ in range(3)]

        self.transforms = {
            'train': Compose(
                [
                    RandomAffine(degrees=(0, 0), translate=(0.05, 0.05)),
                    Normalize(mean=[self._mean[0]], std=[self._std[0]]),
                ]
            ) if augment else Normalize(mean=[self._mean[0]], std=[self._std[0]]),
            'val': Compose(
                [
                    RandomAffine(degrees=(0, 0), translate=(0.05, 0.05)),
                    Normalize(mean=[self._mean[1]], std=[self._std[1]]),
                ]
            ) if augment else Normalize(mean=[self._mean[1]], std=[self._std[1]]),
            'test': Normalize(mean=[self._mean[2]], std=[self._std[2]])
        }
        if train_range:
            train_range = partial(self.filter, range=train_range)

        super().__init__(
            training_file=train,
            validation_file=val,
            prediction_file=test,
            testing_file=test,
            config_file=config,
            budget=budget,
            batch_size=batch_size,
            load_to_memory=False,
            pin_memory=True,
            shuffle=True,
            num_worker=12,
            persistent_workers=True,
            filter_fnc=train_range
        )


    @staticmethod
    def norm_fn(sample, fn):
        return sample[0], fn(sample[1])

    @staticmethod
    def filter(sample, range):
        low, high = range
        return torch.where((sample[:, 0] > low) * (sample[:, 0] <= high), 1, 0)

    def setup(self, stage: str):
        if self._train_dataloader is None:
            self._train_dataloader = H5Dataset(
                self._training_file.resolve().__str__(), ['central', '0/x'],
                budget=self._budget,
                load_to_memory=self.load_to_memory,
                theta_mask=[self._theta_mask, slice(None)],
                transforms=partial(self.norm_fn, fn=self.transforms['train']),
                filter_label='central',
                filter_function=self._filter_fnc,
            )
        if self.split_train is not None:
            self._train_dataloader, self._val_dataloader = torch.utils.data.random_split(self._train_dataloader,
                                                                                         [self.split_train, 1-self.split_train])
            # Set test==val s.t. plots are based on validation set
            self._test_dataloader = self._val_dataloader

        if self._reference_sample is None and self._prediction_file is not None:
            with H5File(self._prediction_file.resolve().__str__(), "r") as _f:
                self._reference_sample = torch.as_tensor(_f['central'][0][self._theta_mask], dtype=torch.float).unsqueeze(0)
        if stage == "predict" and self._prediction_file is not None:
            self._pred_dataloader = H5Dataset(
                self._prediction_file.resolve().__str__(),
                ['0/x'],
                transforms=partial(self.norm_fn, fn=self.transforms['test']),
                filter_label='central',
            )
        elif (stage == "validate" or stage == "fit") and self.split_train is None:
            self._val_dataloader = H5Dataset(
                self._validation_file.resolve().__str__(),
                ['central', '0/x'],
                theta_mask=[self._theta_mask, slice(None)],
                load_to_memory=self.load_to_memory,
                transforms=partial(self.norm_fn, fn=self.transforms['val']),
                filter_label='central',
            ) if self._validation_file is not None else None
        elif stage == "test" and self.split_train is None:
            self._test_dataloader = H5Dataset(
                self._testing_file.resolve().__str__(),
                ['central', '0/x'],
                theta_mask=[self._theta_mask, slice(None)],
                load_to_memory=self.load_to_memory,
                transforms=partial(self.norm_fn, fn=self.transforms['test']),
                filter_label='central',
                # transform_fn=self.transforms['test'],
            ) if self._testing_file is not None else None

        if 'cuda' in str(self.device) and not self.pin_memory:
            import warnings
            warnings.warn(f"Consider pin_memory=True because current device is {self.device} (GPU) and therefore it can cause problems or slow downs.")


class H5DataModuleStatic_Exp(H5DataModuleStatic):
    def __init__(self, training_file: Path, norm_exp = False, **kwargs):
        super().__init__(training_file, **kwargs)
        sys.path.append(r"C:\Users\dwolf\PycharmProjects\DeepLearning_DHM_Correction")
        self.norm_exp = norm_exp

    def setup(self, stage: str):
        if self._train_dataloader is None:
            self._train_dataloader = H5Dataset(
                self._training_file.resolve().__str__(), ['central', '0/x'],
                budget=self._budget,
                load_to_memory=self.load_to_memory,
                theta_mask=[self._theta_mask, slice(None)],
                transforms=self.normalize_transform,
                transform_fn=self.transform_fn,
            )
        if self._reference_sample is None:
            with H5File(self._prediction_file.resolve().__str__(), "r") as _f:
                self._reference_sample = torch.as_tensor(_f['central'][0][self._theta_mask],
                                                         dtype=torch.float).unsqueeze(0)
        if stage == "predict" and self._prediction_file is not None:
            self._pred_dataloader = H5Dataset(
                self._prediction_file.resolve().__str__(),
                ['0/x'],
                transforms=self.normalize_transform,
                transform_fn=self.transform_fn,
            )
        elif stage == "validate" or stage == "fit":
            exp_module = DHMexpModuleStatic(data_size=256, normalize=self.norm_exp)
            exp_module.setup('fit')
            self._val_dataloader = exp_module._dataloader if self._validation_file is not None else None
        elif stage == "test":
            self._test_dataloader = H5Dataset(
                self._testing_file.resolve().__str__(),
                ['central', '0/x'],
                theta_mask=[self._theta_mask, slice(None)],
                load_to_memory=self.load_to_memory,
                transforms=self.normalize_transform,
                transform_fn=self.normalize_fn,
            ) if self._testing_file is not None else None

        if 'cuda' in str(self.device) and not self.pin_memory:
            import warnings
            warnings.warn(
                f"Consider pin_memory=True because current device is {self.device} (GPU) and therefore it can cause problems or slow downs.")

    def val_dataloader(self):
        return DataLoader(self._val_dataloader,
                          batch_size=self.batch_size,
                          num_workers=self.num_worker,
                          shuffle=False,
                          pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers,
                          worker_init_fn=self.worker_init,
                          ) if self._val_dataloader is not None else None

