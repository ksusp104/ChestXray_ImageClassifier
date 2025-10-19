from torchvision import transforms #pytorch module for image transformations

#creates a sequence of image preprocessing steps to prepare model for training/evaluation
def make_transforms(input_size=(224,224), use_imagenet_norm=True, train=False):
    #start with an empty list of transformations
    t = []
    #add data augmentation transforms (training specific will likely need to come back)
    if train:
        #flips image horizontaly and rotates image slightly (may not keep this in)
        t += [transforms.RandomHorizontalFlip(p=0.5),
              transforms.RandomRotation(degrees=5)]
        #scale the image to input size model expects
        t += [transforms.Resize(input_size, antialias=True),
              transforms.ToTensor()]
        #add normalization (optional)
        if use_imagenet_norm:
            t += [transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])]
        #combine all transformations into a single callable pipeline
        return transforms.Compose(t)
