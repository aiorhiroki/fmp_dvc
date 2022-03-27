import farmer_pytorch as fmp
import segmentation_models_pytorch as smp
import albumentations as albu
import torch


class AnnotationImp(fmp.GetAnnotationABC):
    target = "./seg_data/CamVid"
    get_train_fn = fmp.readers.CaseDirect("train", "trainannot")
    get_val_fn = fmp.readers.CaseDirect("val", "valannot")

    """
    def __call__(self):
        # you can override GetAnnotation function
        return train_set, validation_set
    """


class DatasetImp(fmp.GetDatasetSgmABC):
    class_values = [8]
    train_trans = albu.Compose(
        [albu.augmentations.geometric.resize.Resize(256, 512),
         albu.augmentations.transforms.HorizontalFlip(p=0.5)])
    val_trans = albu.Compose(
        [albu.augmentations.geometric.resize.Resize(256, 512)])

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


class OptimizationImp(fmp.GetOptimizationABC):
    batch_size = 16
    epochs = 3
    lr = 0.001
    gpu = 2
    optimizer = torch.optim.Adam
    model = smp.FPN(encoder_name="efficientnet-b7", encoder_weights="imagenet",
                    activation="sigmoid", in_channels=3, classes=1,)
    loss_func = smp.losses.DiceLoss('binary', from_logits=False)
    metric_func = fmp.metrics.Dice()

    """
    def on_epoch_end(self):
        # set custom callbacks
    """


def command():
    train_anno, val_anno = AnnotationImp()()
    train, val = DatasetImp(train_anno, training=True), DatasetImp(val_anno)
    OptimizationImp(train, val)()


if __name__ == "__main__":
    command()
