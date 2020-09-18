from torchvision.datasets.vision import VisionDataset

from PIL import Image

import os
import os.path
import numpy as np 
from albumentations import (
    Compose,
    OneOf,
    RandomBrightness,
    RandomContrast,
    IAAAdditiveGaussianNoise,
    GaussNoise,
    HorizontalFlip,
    VerticalFlip
    
)
import random
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(directory1, directory2, class_to_idx, extensions=None, is_valid_file=None):
    instances = []
    directory1 = os.path.expanduser(directory1)
    directory2 = os.path.expanduser(directory2)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()): #the target_class should be same in dir1 and dir2.
        class_index = class_to_idx[target_class]
        target_dir1 = os.path.join(directory1, target_class)
        target_dir2 = os.path.join(directory2, target_class)
        if not os.path.isdir(target_dir1) or not os.path.isdir(target_dir2):
            continue
        for root, _, fnames in sorted(os.walk(target_dir1, followlinks=True)):
            for fname in sorted(fnames):
                fname2 = fname.replace('-', '+')
                path1 = os.path.join(root, fname)
                path2 = os.path.join(target_dir2, fname2)
                if is_valid_file(path1) and is_valid_file(path2):
                    item = path1, path2, class_index
                    instances.append(item)
    return instances

def aug_image(img):
    aug = Compose(
        [
        OneOf([RandomBrightness(limit=0.1, p=0.3), RandomContrast(limit=0.1, p=0.3)]),
        #IAAAdditiveGaussianNoise(p=0.3)
        ]
    )
    img_aug = aug(image=img)['image']
    return img_aug

class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root1, root2, loader, extensions=None, transform1=None,
                 transform2=None,target_transform=None, is_valid_file=None, aug = False):
        super(DatasetFolder, self).__init__(root1, transform=transform1,
                                            target_transform=target_transform)
        self.root1 = root1
        self.root2 = root2
        self.transform1 = transform1
        self.transform2 = transform2
        self.aug = aug
        classes, class_to_idx = self._find_classes(self.root1)
        
        samples = make_dataset(self.root1, self.root2, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """ 
        path1, path2, target = self.samples[index]
        img1, img2 = self.loader(path1, path2)

        if self.transform is not None:
            img1 = self.transform1(img1)
            img2 = self.transform2(img2)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        
            
        s1 = np.array(img1)
        s2 = np.array(img2)
        
        if self.aug == True:
            #s1 = aug_image(s1)
            #s2 = aug_image(s2)
            if(0.1*random.randint(0,9)<0.6):
                s1 = HorizontalFlip(p=1)(image=s1)['image']
                s2 = HorizontalFlip(p=1)(image=s2)['image']
            if(0.1*random.randint(0,9)<0.6):
                s1 = VerticalFlip(p=1)(image=s1)['image']
                s2 = VerticalFlip(p=1)(image=s1)['image']
        #sample = np.concatenate((s1, s2), axis=0)
        sample = s1*0.4 + s2*0.6
        
        return sample, target

    def __len__(self):
        return len(self.samples)


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path1, path2):
    img1 = Image.open(path1)
    img2 = Image.open(path2)
    return img1.convert('RGB'), img2.convert('RGB')


def accimage_loader(path1, path2):
    import accimage
    try:
        return accimage.Image(path1), accimage.Image(path2)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path1, path2)


def default_loader(path1, path2):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path1, path2)
    else:
        return pil_loader(path1, path2)


class ImageFolder2(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, root1, root2, transform1=None, transform2=None, target_transform=None,
                 loader=default_loader, is_valid_file=None, aug = False):
        super(ImageFolder2, self).__init__(root1, root2, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform1=transform1,
                                          transform2=transform2,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file, aug = aug)
        self.imgs = self.samples