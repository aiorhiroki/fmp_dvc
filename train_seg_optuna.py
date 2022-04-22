import farmer_pytorch as fmp
import segmentation_models_pytorch as smp
import albumentations as albu
import torch

import argparse
import optuna
import os
import pandas as pd

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

def parse_args():
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
    parser.add_argument(
        "--model_name", dest="model_name", default=None, type=str)
    parser.add_argument(
        "--dataset_name", dest="dataset_name", default=None, type=str)
    parser.add_argument(
        "--batch_size", dest="batch_size", default=None, type=int)
    parser.add_argument(
        "--gpus", dest="gpus", default=None, type=str)
    parser.add_argument(
        "--result_dir", dest="result_dir", default=None, type=str
    )

    return parser.parse_args()

class Objective(object):
    use_optuna = True
    # dataset
    target = "./seg_data/CamVid"
    get_train_fn = fmp.readers.CaseDirect("train", "trainannot")
    get_val_fn = fmp.readers.CaseDirect("val", "valannot")
    class_values = [1, 8]
    train_trans = albu.Compose(
        [albu.augmentations.geometric.resize.Resize(256, 512),
            albu.augmentations.transforms.HorizontalFlip(p=0.5)])
    val_trans = albu.Compose(
        [albu.augmentations.geometric.resize.Resize(256, 512)])
    # epochs
    epochs = 30
    # optimizer
    optimizer_name_list = ["RAdam", "Adam", "RMSprop", "Adadelta", "AdamW", "SGD"]
    # lr
    lr_range = [1e-5, 1e-2]
    # loss_func
    loss_name_list = ["Diceloss", "JaccardLoss", "TverskyLoss", "LovaszLoss"]
    # models
    
    def __init__(self, args):
        self.model_name = args.model_name
        self.batch_size = args.batch_size
        self.gpus = args.gpus
        self.result_dir = args.result_dir 

    def __call__(self, trial):
        self.choose_optim(trial)
        self.choose_lr(trial)
        self.choose_loss_func(trial)
        self.choose_model(trial)

        train_anno, val_anno = AnnotationImp(
        self.target, get_train_fn=self.get_train_fn, get_val_fn=self.get_val_fn)()
        train = DatasetImp(train_anno, self.class_values, self.train_trans)
        val = DatasetImp(val_anno, self.class_values, self.val_trans)
        print(f"\ntrial number:[{trial.number}], Params: {trial.params}\n")

        
        return OptimizationImp(
            train, val,
            self.batch_size, 
            self.epochs, 
            self.lr, 
            self.gpus, 
            self.optimizer_cls, 
            self.model, 
            self.loss_func, 
            self.result_dir, 
            self.use_optuna)(trial)


    def data_trans(self):
        # data augumentation
        pass

    def choose_optim(self, trial):
        optimizer_name = trial.suggest_categorical("optimizer", self.optimizer_name_list)
        if optimizer_name == self.optimizer_name_list[0]:
            self.optimizer_cls = torch.optim.RAdam
        elif optimizer_name == self.optimizer_name_list[1]:
            self.optimizer_cls = torch.optim.Adam
        elif optimizer_name == self.optimizer_name_list[2]:
            self.optimizer_cls = torch.optim.RMSprop
        elif optimizer_name == self.optimizer_name_list[3]:
            self.optimizer_cls = torch.optim.Adadelta
        elif optimizer_name == self.optimizer_name_list[4]:
            self.optimizer_cls = torch.optim.AdamW
        elif optimizer_name == self.optimizer_name_list[5]:
            self.optimizer_cls = torch.optim.SGD
    
    def choose_lr(self, trial):
        self.lr = trial.suggest_float("learning_rate", self.lr_range[0], self.lr_range[1], log=True)
    
    def choose_loss_func(self, trial):
        loss_name = trial.suggest_categorical("loss_func", self.loss_name_list)
        if loss_name == self.loss_name_list[0]:
            self.loss_func = smp.losses.DiceLoss('multilabel', from_logits=False)
        elif loss_name == self.loss_name_list[1]:
            self.loss_func = smp.losses.JaccardLoss('multilabel', from_logits=False)
        elif loss_name == self.loss_name_list[2]:
            self.loss_func = smp.losses.TverskyLoss('multilabel', from_logits=False, alpha = 0.3, beta = 0.7)
        elif loss_name == self.loss_name_list[3]:
            self.loss_func = smp.losses.LovaszLoss('multilabel')

    def choose_model(self, trial):
        if self.model_name == "deeplabv3p":
        # for deeplabv3p parameter
            encoder_name_list = ["efficientnet-b7", "timm-regnety_320", "resnet152", "tu-xception71", "resnext101_32x48d", "timm-efficientnet-l2"]

            encoder_name = trial.suggest_categorical("encoder_name", encoder_name_list)
            if encoder_name == encoder_name_list[4]:
                weights = "instagram"
            elif encoder_name == encoder_name_list[5]:
                weights = "noisy-student"
            else:
                weights = "imagenet"

            self.model = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_depth=5,
                                    encoder_weights=weights, encoder_output_stride=16,
                                    decoder_channels=256, decoder_atrous_rates=(21, 24, 36),
                                    in_channels=3, classes=2, activation="sigmoid", upsampling=4, aux_params=None)

        elif self.model_name == "FPN":
            # for FPN parameter
            encoder_name_list = ["efficientnet-b7", "timm-regnety_320", "resnet152", "tu-xception71", "resnext101_32x48d", "timm-efficientnet-l2"]

            encoder_name = trial.suggest_categorical("encoder_name", encoder_name_list)
            if encoder_name == encoder_name_list[4]:
                weights = "instagram"
            elif encoder_name == encoder_name_list[5]:
                weights = "noisy-student"
            else:
                weights = "imagenet"

            self.model = smp.FPN(encoder_name=encoder_name, 
                            encoder_depth=5, 
                            encoder_weights=weights, 
                            decoder_pyramid_channels=256, 
                            decoder_segmentation_channels=128, 
                            decoder_merge_policy='add', 
                            decoder_dropout=0.2, 
                            in_channels=3, 
                            classes=2, 
                            activation="sigmoid", 
                            upsampling=4, 
                            aux_params=None)

        elif self.model_name == "Linknet":
            encoder_name_list = ["efficientnet-b7", "timm-regnety_320", "resnet152", "tu-xception71", "resnext101_32x48d", "timm-efficientnet-l2"]

            encoder_name = trial.suggest_categorical("encoder_name", encoder_name_list)
            if encoder_name == encoder_name_list[4]:
                weights = "instagram"
            elif encoder_name == encoder_name_list[5]:
                weights = "noisy-student"
            else:
                weights = "imagenet"

            self.model = smp.Linknet(encoder_name=encoder_name, 
                                encoder_depth=5, 
                                encoder_weights=weights, 
                                decoder_use_batchnorm=True, 
                                in_channels=3, 
                                classes=2, 
                                activation="sigmoid", 
                                aux_params=None)

        elif self.model_name == "MAnet":
            encoder_name_list = ["efficientnet-b7", "timm-regnety_320", "resnet152", "tu-xception71", "resnext101_32x48d", "timm-efficientnet-l2"]

            encoder_name = trial.suggest_categorical("encoder_name", encoder_name_list)
            if encoder_name == encoder_name_list[4]:
                weights = "instagram"
            elif encoder_name == encoder_name_list[5]:
                weights = "noisy-student"
            else:
                weights = "imagenet"

            self.model = smp.MAnet(encoder_name=encoder_name, 
                              encoder_depth=5, 
                              encoder_weights=weights, 
                              decoder_use_batchnorm=True, 
                              decoder_channels=(256, 128, 64, 32, 16), 
                              decoder_pab_channels=64, 
                              in_channels=3, 
                              classes=2, 
                              activation="sigmoid", 
                              aux_params=None)

def main(args):
    # 学習を中断しても再開できるようにDBに保存
    study_name = args.model_name + "_" + args.dataset_name
    storage = f'sqlite:///{study_name}_output_database.db'

    objective = Objective(args)

    study = optuna.create_study(sampler=optuna.samplers.TPESampler(),
                                pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource='auto', reduction_factor=4, min_early_stopping_rate=0),
                                direction = "maximize", 
                                study_name=study_name,
                                storage=storage,
                                load_if_exists=True)
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best params:", study.best_params)
    print("Best trial:", study.best_trial)
    print("  Value: {}".format(study.best_trial.value))
    best_params_file = f"{study_name}_best_params.csv"
    if os.path.isfile(best_params_file):
        number_and_value = {'number': study.best_trial.number, 'Value': study.best_trial.value}
        df = pd.json_normalize({**study.best_params, **number_and_value})
        df.to_csv(best_params_file, mode='a', index=False)
    else: 
        number_and_value = {'number': study.best_trial.number, 'Value': study.best_trial.value}
        df = pd.json_normalize({**study.best_params, **number_and_value})
        df.to_csv(best_params_file)
    
if __name__ == "__main__":
    args = parse_args()
    print(args)
    print(args.model_name, "_", args.dataset_name)
    main(args)

