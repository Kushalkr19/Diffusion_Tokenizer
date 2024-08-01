import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split
import numpy as np
import pickle
from itertools import repeat
import os
from typing import Dict, List, Tuple, Union

class CubeData(Dataset):
    """Cube Dataset"""
    def __init__(self, root: str, name: str, transform=None, folder=True, file=None) -> None:
        """Create a dataset from a given directory and pickle file."""
        super().__init__()
        self.transform = transform
        self.root = root
        self.name = name
        if folder:
            self.cubeFileNames = [f for f in os.listdir(os.path.join(self.root, name, "python")) if ".npy" in f]
        else:
            self.cubeFileNames = [file] #os.path.join(self.root, name, "python", file)
        if len(self.cubeFileNames) == 0:
            raise ValueError("Couldn't find any numpy files in given folder!")
        
        self.cubes = self.__load_cubes()
        if not os.path.exists(os.path.join(root, f"selected_patches/{self.name}.pkl")):
            raise ValueError(f"Selected Patches file not found in {os.path.join(root, f'selected_patches/{self.name}.pkl')}")
        with open(os.path.join(root, f"selected_patches/{self.name}.pkl"), 'rb') as f:
            selected_patches = pickle.load(f)
        self.selected_patches = {k.replace('.npy',''): v for k,v in selected_patches.items()}
        self.lookupIdx = self.__create_idx_lookup_table()

    def __load_cubes_old(self) -> Dict[str, np.ndarray]:
        """Loads all cubes contained in the folder and stores them in a dictionary indexed by filename."""
        cubes = {}
        total_size = 0
        for file in self.cubeFileNames:
            current_cube = np.load(os.path.join(self.root, self.name, "python", file))
            fileName = file.split('.')[0]
            cubes[fileName] = current_cube[5:77]
            total_size += current_cube[5:77].nbytes
        print(f"Total size of loaded cubes: {total_size / (1024**3):.2f} GB")
        return cubes
    
    def __load_cubes_m3(self) -> Dict[str, np.ndarray]:
        """Loads all cubes contained in the folder and stores them in a dictionary indexed by filename."""
        cubes = {}
        total_size = 0
        for file in self.cubeFileNames:
            file_path = os.path.join(self.root, self.name, "python", file)
            current_cube = np.load(file_path, mmap_mode='r')  # Memory-map the file
            fileName = file.split('.')[0]
            cubes[fileName] = current_cube[5:77]
            total_size += current_cube[5:77].nbytes

        print(f"Total size of loaded cubes: {total_size / (1024**3):.2f} GB")
        return cubes
    
    def __load_and_reshape(self,file_path):
        # Load the data
        current_cube = np.load(file_path)#, mmap_mode='r
        
        # Check if the array is 2D
        if current_cube.ndim == 2:
            # Add a new axis at the beginning (axis=0)
            current_cube = np.expand_dims(current_cube, axis=0).astype(np.float32)
    
        return current_cube
    
    def __load_cubes(self) -> Dict[str, np.ndarray]:
        """Loads all cubes contained in the folder and stores them in a dictionary indexed by filename."""
        cubes = {}
        total_size = 0
        for file in self.cubeFileNames:
            file_path = os.path.join(self.root, self.name, "python", file)
            current_cube = self.__load_and_reshape(file_path)  # Memory-map the file
            fileName = file.split('.')[0]
            cubes[fileName] = current_cube
            total_size += current_cube.nbytes

        print(f"Total size of loaded cubes: {total_size / (1024**3):.2f} GB")
        return cubes

    def __create_idx_lookup_table(self) -> List[Tuple[str, int]]:
        """Creates a lookup table of form: [('file1', 0), ('file1', 1), ('file1', 2), ... ('file3', 42)]"""
        lookup = []
        for file in self.cubeFileNames:
            fileName = file.split('.')[0]
            #print(self.selected_patches.keys())
            #print(fileName)
            len_patches = self.selected_patches[fileName].shape[0]
            lookup += list(zip(repeat(fileName), range(0, len_patches)))
        return lookup

    def __len__(self) -> int:
        return sum(self.selected_patches[file.split('.')[0]].shape[0] for file in self.cubeFileNames)
    
    def __getitem__(self, index: Union[int, List[int], slice]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(index, (torch.Tensor, np.ndarray)):
            index = index.tolist()

        if isinstance(index, int):
           
            fileName, localIdx = self.lookupIdx[index]
            xmin, xmax, ymin, ymax = self.selected_patches[fileName][localIdx, :]
            selectedPatches = torch.from_numpy(self.cubes[fileName][:, ymin:ymax, xmin:xmax].copy()) #.copy()
            
        elif isinstance(index, list):
            selectedPatches = []
            for cIndex in index:
                fileName, localIdx = self.lookupIdx[cIndex]
                xmin, xmax, ymin, ymax = self.selected_patches[fileName][localIdx, :]
                selectedPatches.append(torch.from_numpy(self.cubes[fileName][:, ymin:ymax, xmin:xmax].copy()))
            selectedPatches = torch.stack(selectedPatches)
        elif isinstance(index, slice):
            selectedPatches = []
            step = index.step if index.step else 1
            for i in range(index.start, index.stop, step):
                fileName, localIdx = self.lookupIdx[i]
                xmin, xmax, ymin, ymax = self.selected_patches[fileName][localIdx, :]
                selectedPatches.append(torch.from_numpy(self.cubes[fileName][:, ymin:ymax, xmin:xmax].copy()))
            selectedPatches = torch.stack(selectedPatches)

        if self.transform:
            selectedPatches = self.transform(selectedPatches)

        mask = self.__computeMask(selectedPatches).to(torch.bool)
        selectedPatches = torch.nan_to_num(selectedPatches, nan=0.0)
        
        return selectedPatches, mask
    
    def __computeMask(self, img: torch.Tensor) -> torch.Tensor:
        """Computes a binary mask for a given image / cube."""
        mask = torch.zeros(img.shape)
        mask[torch.where(img == 0)] = 1.0

        if torch.isnan(img).any():
            mask[torch.isnan(img)] = 1.0

        if len(img.shape) == 3:
            c, h, w = img.shape
            summed = torch.sum(mask, dim=0)
            mask = torch.where(summed == c, 1.0, 0.0)[None, ...]
        else: 
            n, c, h, w = img.shape
            summed = torch.sum(mask, dim=1)
            mask = torch.where(summed == c, 1.0, 0.0)[:, None, ...]

        return mask

if __name__ == "__main__":
    data = CubeData("/home/tejaspanambur/fdl-2024-lunar/H3Tokenizer/data/", "geochemical_maps", folder=False, file='Global20ppd_LPGRS_geotiff_Ti.npy')
    print(f"len(data)={len(data)}")

    someCube, someMask = data[42]
    someSlicedCubes, someSlicedMasks = data[10:42]
    someOtherCubes, someOtherMasks = data[[0, 2, 5, 6]]
    train, test = random_split(data, (0.8, 0.2))
    test_loader = DataLoader(train, batch_size=2, shuffle=True)
    img, mask = next(iter(test_loader))
    print(img.shape)
    print(mask.shape)
    print(img.dtype)

    print("finished!")