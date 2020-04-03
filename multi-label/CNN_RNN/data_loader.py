import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import json
import numpy as np
import nltk
from PIL import Image
from build_vocab import Vocabulary
from pycocotools.coco import COCO


class CocoDataset(data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, idx2path, my_data, vocab, transform=None, new_dataset=True):
        """Set the path for images, captions and vocabulary wrapper.
        
        Args:
            root: image directory.
            json: coco annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        #self.coco = COCO(json)
        #self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform
        with open(my_data, 'r') as infile:
            self.MyData = json.load(infile)
        with open(idx2path, 'r') as infile:
            self.idx2path = json.load(infile)
        if new_dataset:
            mypath = os.path.join('..','..','..','data_tate')
            onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            MyData1 = {}
            idx2path1 = {}
            for i,img_targ in enumerate(self.MyData.keys()):
            if img_targ in onlyfiles:
                MyData1[img_targ] = self.MyData[img_targ]
                idx2path1[str(i)] = self.idx2path[str(i)]
            self.MyData = MyData1
            self.idx2path = idx2path1
          
        
    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        #coco = self.coco
        vocab = self.vocab
        #ann_id = self.ids[index]
        #caption = coco.anns[ann_id]['caption']
        #img_id = coco.anns[ann_id]['image_id']
        #path = coco.loadImgs(img_id)[0]['file_name']
        path = self.idx2path[str(index)]
        caption_ = self.MyData[path]
        image = Image.open(os.path.join(self.root, path)).resize((224, 224)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = caption_ #nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.MyData)


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).
    
    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return images, targets, lengths

def get_loader(root, idx2path, my_data, vocab, batch_size, shuffle, transform, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       idx2path=idx2path,
                       my_data=my_data,
                       vocab=vocab,
                       transform=transform)
    
    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader