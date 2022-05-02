import re
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
    parser = argparse.ArgumentParser(description='To search model with optuna')
    # params of training
    parser.add_argument(
        "--study_name", dest="study_name", default=None, type=str)
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
    optimizer_list = [
    "torch.optim.RAdam", "torch.optim.Adam", "torch.optim.RMSprop",
    "torch.optim.Adadelta", "torch.optim.AdamW", "torch.optim.SGD"]
    # lr
    lr_range = [1e-5, 1e-2]
    # loss_func
    default_loss_func_args = dict(mode='multilabel', from_logits=False)
    loss_func_list = [
            dict(loss_func_name="smp.losses.DiceLoss", args=default_loss_func_args),
            dict(loss_func_name="smp.losses.JaccardLoss", args=default_loss_func_args),
            dict(loss_func_name="smp.losses.TverskyLoss", args=default_loss_func_args | dict(alpha = 0.3, beta = 0.7))
        ]
    # models
    default_args = dict(classes=2, activation="softmax")
    default_encoders = ["efficientnet-b7", "timm-regnety_320", "resnet152",
                        "resnext101_32x48d", "timm-efficientnet-l2"]

    model_list = [
        dict(model="smp.DeepLabV3Plus", args=default_args, encoders=default_encoders),
        dict(model="smp.FPN", args=default_args, encoders=default_encoders + ["tu-xception71"]),
        dict(model="smp.Linknet", args=default_args, encoders=default_encoders),
        dict(model="smp.MAnet", args=default_args | dict(encoder_depth=5), encoders=default_encoders)
        ]
    
    def __init__(self, args):
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
        try:
            result = OptimizationImp(
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
            return result

        except RuntimeError as e:
            print(e)
            if self.batch_size != 1:
                self.batch_size = int(self.batch_size / 2)
                print("batch_size:", self.batch_size)

            return 0


    def data_trans(self, trial):
        # data augumentation
        pass
            
    def choose_optim(self, trial):
        optimizer_name = trial.suggest_categorical("optimizer", self.optimizer_list)
        self.optimizer_cls = eval(optimizer_name)
    
    def choose_lr(self, trial):
        self.lr = trial.suggest_float("learning_rate", self.lr_range[0], self.lr_range[1], log=True)
    
    def choose_loss_func(self, trial):
        loss_func_num = trial.suggest_int("loss_func_num", 0, len(self.loss_func_list) - 1)
        self.loss_func = eval(self.loss_func_list[loss_func_num]['loss_func_name']
                              + "(**self.loss_func_list[loss_func_num]['args'])")

    def choose_model(self, trial):
        model_num = trial.suggest_int("model_num", 0, len(self.model_list) - 1)
        encoder_num = trial.suggest_int("encoder_num", 0, len(self.model_list[model_num]['encoders']) - 1)
        encoder_name = self.model_list[model_num]['encoders'][encoder_num]
        encoder_weight = {"resnext101_32x48d": "instagram", "timm-efficientnet-l2": "noisy-student"}
        weights = encoder_weight.get(encoder_name) or "imagenet"
        
        self.model = eval(self.model_list[model_num]['model']
                            + "(**self.model_list[model_num]['args'], encoder_name=encoder_name, encoder_weights=weights)")

def main(args):
    # store study params to the Data base in order to restart study even if study was interrupted
    study_name = args.study_name
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
    main(args)

