import torch
from astropy.io import fits
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from astropy.visualization import SqrtStretch, MinMaxInterval
import numpy as np
import os

class psf_dataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.size = len(os.listdir(root_dir))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, id):

        if id >= self.size:
            raise ValueError('[Dataset] Index out of bounds')
            return None

        sample_name = self.root_dir + 'psf_' + str(int(id)) + '.fits'
        sample_hdu = fits.open(sample_name)
        zernike = np.float32(sample_hdu[0].data)
        image = np.float32(sample_hdu[2].data)
        #image = np.stack((sample_hdu[2].data, sample_hdu[3].data)).astype(np.float32)

        phase = np.float32(sample_hdu[1].data)

        sample = {'phase': phase, 'image': image , 'zernike' : zernike, "name": sample_name}

        if self.transform:
            """ sample['phase'] = self.transform(sample['phase'])
            sample['image'] = self.transform(sample['image']) """

            sample = self.transform(sample)
        return sample


class Normalize(object):
    def __call__(self, sample):
        phase, image = sample['phase'], sample['image']
        
        """ image[0] = minmax(np.sqrt(image[0]))
        image[1] = minmax(np.sqrt(image[1])) """

        image = minmax(np.sqrt(image))
        
        #phase = (phase/810.)*2*np.pi

        return {'phase': phase, 'image': image, 'zernike': sample['zernike']}

    
def minmax(array):
    a_min = np.min(array)
    a_max = np.max(array)
    return (array-a_min)/(a_max-a_min)    

class ToTensor(object):
    def __call__(self, sample):
        phase, image = sample['phase'], sample['image']
        zer = sample['zernike']

        return {'phase': torch.from_numpy(np.float32(phase)).unsqueeze(0), 'image': torch.from_numpy(np.float32(image)).unsqueeze(0), 'zernike' : torch.from_numpy(np.float32(zer))}

class Noise(object):
    def __call__(self, sample):
        phase, image = sample['phase'], sample['image']
        
        noise_intensity = 1000
        image = minmax(image)
        #image[1] = minmax(image[1])
        image = np.random.poisson(lam=noise_intensity*image, size=None)
        #image[1] = np.random.poisson(lam=noise_intensity*image[1], size=None)

        return {'phase': phase, 'image': image, 'zernike':  sample['zernike']}


def splitDataLoader(dataset, split=[0.8, 0.2], batch_size=32, random_seed=None, shuffle=True):
    indices = list(range(len(dataset)))
    s = int(np.floor(split[1] * len(dataset)))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[s:], indices[:s]

    train_sampler, val_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=train_sampler)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=val_sampler)

    return train_dataloader, val_dataloader
