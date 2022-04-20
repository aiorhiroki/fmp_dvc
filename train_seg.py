import farmer_pytorch as fmp
import segmentation_models_pytorch as smp
import albumentations as albu
import torch


class AnnotationImp(fmp.GetAnnotation):
    """
    def __call__(self):
        # you can override GetAnnotation function
        return train_set, validation_set
    """


class DatasetImp(fmp.GetDatasetSgm):
    """
    def preprocess(self, image, mask):
        # set custom preprocessing function
        return image, mask
    """

    """
    def __getitem__(self, i):
        # you can override getitem function
        # use self.annotation, self.augmentation ...
        return in, out
    """


class OptimizationImp(fmp.GetOptimization):
    """
    def set_scheduler(self):
        self.scheduler = ...
    """


def main():
    target = "./seg_data/CamVid"
    get_train_fn = fmp.readers.CaseDirect("train", "trainannot")
    get_val_fn = fmp.readers.CaseDirect("val", "valannot")
    class_values = [1, 8]
    train_trans = albu.Compose(
        [albu.augmentations.geometric.resize.Resize(256, 512),
            albu.augmentations.transforms.HorizontalFlip(p=0.5)])
    val_trans = albu.Compose(
        [albu.augmentations.geometric.resize.Resize(256, 512)])

    batch_size = 16
    epochs = 30
    lr = 0.001
    gpus = "2,3"
    optimizer_cls = torch.optim.Adam
    model = smp.FPN(encoder_name="efficientnet-b7", encoder_weights="imagenet",
                    activation="softmax", in_channels=3, classes=2,)
    loss_func = smp.losses.DiceLoss('multilabel', from_logits=False)

    train_anno, val_anno = AnnotationImp(
        target, get_train_fn=get_train_fn, get_val_fn=get_val_fn)()
    train = DatasetImp(train_anno, class_values, train_trans)
    val = DatasetImp(val_anno, class_values, val_trans)
    OptimizationImp(
        train, val,
        batch_size, epochs, lr, gpus, optimizer_cls, model, loss_func)()


if __name__ == "__main__":
    main()
