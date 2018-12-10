import torch
from torchvision import datasets, transforms


def inception_preproccess(input_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.2)), # move center, adjust ratio, scaling
        transforms.RandomAffine(30, translate=(0.01, 0.01), shear=30, resample=False,
                                            fillcolor=0), # move center, rotation, shearing
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(**normalize)
    ])


def scale_crop(input_size, scale_size=None):
    t_list = [
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        # transforms.Normalize(**normalize),
    ]
    # if scale_size != input_size:
    #     t_list = [transforms.Resize(scale_size)] + t_list

    return transforms.Compose(t_list)


def get_transform(augment=False, input_size=224):
    # normalize = __imagenet_stats
    scale_size = int(input_size / 0.875)
    if augment:
        return inception_preproccess(input_size=input_size)
    else:
        return scale_crop(input_size=input_size, scale_size=scale_size)

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # print(path.type())
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


if  __name__ == "__main__":
    # EXAMPLE USAGE:
    # instantiate the dataset and dataloader

    data_dir = "/home/yue/workplace/dataset/oalign-cvpr2016/train"
    dataset = ImageFolderWithPaths(data_dir, transform=get_transform())  # our custom dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    # iterate over data
    for inputs, labels, paths in dataloader:
        # use the above variables freely
        print(paths)


