import os
import torch
import csv
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

# Import model definition
from models.swin_transformer_multi_tasks import *

# Import dataset definition
from data.GastricCarcinoma import *

# Import loss and training utilities
from utils.losses import *
from utils.train import *
import warnings
warnings.filterwarnings('ignore')  # Ignore all warnings for cleaner output

# ==== Configurations ====
DEVICE = 'cuda:0'
CHECKPOINT_DIR = "./checkpoints/"
BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "Transformer.pth")
BATCH_SIZE = 2
LR = 0.0001
MILESTONES = [50, 100, 200]
GAMMA = 0.5

def main():
    # ==== Directory Preparation ====
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print(f"The best model's location is {CHECKPOINT_DIR}.")

    # ==== Data Preparation ====
    img_transform = GastricCarcinomaTransform()
    train_dataset = GastricCarcinoma_Dataset(
        "./image_samples/", train_val_test="train",
        augmentation=img_transform.get_validation_augmentation()
    )
    val_dataset = GastricCarcinoma_Dataset(
        "./image_samples/", train_val_test="val",
        augmentation=img_transform.get_validation_augmentation()
    )
    test_dataset = GastricCarcinoma_Dataset(
        "./image_samples/", train_val_test="test",
        augmentation=img_transform.get_validation_augmentation()
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    # ==== Model, Loss, Optimizer ====
    model = SwinTransformer_MulTasks(img_size=224, window_size=7, in_chans=1, num_classes=2, other_feature_num=8, ape=True)
    model.to(DEVICE)
    loss_fn = Loss_main_sub1_sub2_task()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=0e-5)

    train_epoch = TrainEpoch(model, loss=loss_fn, optimizer=optimizer, device=DEVICE, verbose=True)
    val_epoch = ValidEpoch(model=model, loss=loss_fn, device=DEVICE)

    # ==== Training Loop ====
    best_auc_main = 0
    best_auc_sub1 = 0
    best_auc_sub2 = 0

    for epoch in range(6):
        print(f"Epoch {epoch} ...")
        train_logs = train_epoch.run(train_loader)
        if epoch % 2 == 0:
            val_logs = val_epoch.run(val_loader)
            if best_auc_main < val_logs["main_task"]['AUC']:
                best_auc_main = val_logs["main_task"]['AUC']
                best_auc_sub1 = val_logs["sub1_task"]['AUC']
                best_auc_sub2 = val_logs["sub2_task"]['AUC']
                print("\nUpdate the trained model and best validation performance...")
                print("{:>4s}\t{:>4s}\t{:>4s}\t{:>4s}\t{:>4s}\t{:>4s}".format(
                    "Train", "Sub1", "Sub2", "Validation", "Sub1", "Sub2"))
                print("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(
                    train_logs["main_task"]["AUC"], train_logs["sub1_task"]["AUC"], train_logs["sub2_task"]["AUC"],
                    val_logs["main_task"]["AUC"], val_logs["sub1_task"]["AUC"], val_logs["sub2_task"]["AUC"]))
                print("Best Sub1 AUC: {:.3f}\tBest Sub2 AUC: {:.3f}".format(best_auc_sub1, best_auc_sub2))
                torch.save(model.state_dict(), BEST_MODEL_PATH)

    # ==== Evaluation ====
    model = SwinTransformer_MulTasks(img_size=224, window_size=7, in_chans=1, num_classes=2, other_feature_num=8, ape=True)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    test_epoch = ValidEpoch(model=model, loss=loss_fn, device=DEVICE)
    logs_final_test = test_epoch.run(test_loader)
    logs_final_validation = val_epoch.run(val_loader)

    # ==== Result Output ====
    with open(BEST_MODEL_PATH + "results.csv", 'w+', newline='') as csv_file:
        results_writer = csv.writer(csv_file)
        results_writer.writerow(['target_main', 'predicted_main', 'prob_main', 'target_sub1', 'predicted_sub1', 'prob_sub1', 'target_sub2', 'predicted_sub2', 'prob_sub2'])
        for i in range(len(logs_final_validation["main_task"]["Targets"])):
            results_writer.writerow([
                logs_final_validation["main_task"]["Targets"][i], logs_final_validation["main_task"]["Probs_label"][i], logs_final_validation["main_task"]["Probs"][i],
                logs_final_validation["sub1_task"]["Targets"][i], logs_final_validation["sub1_task"]["Probs_label"][i], logs_final_validation["sub1_task"]["Probs"][i],
                logs_final_validation["sub2_task"]["Targets"][i], logs_final_validation["sub2_task"]["Probs_label"][i], logs_final_validation["sub2_task"]["Probs"][i]
            ])

    print("-------------------------Validation Results--------------------------")
    print("{:>4s}\t{:>4s}\t{:>4s}\t{:>4s}\t{:>4s}\t{:>4s}".format(
        "AUC", "ACC", "PPV", "Sensitivity", "Fscore", "Sensitivity"))
    print("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(
        logs_final_validation["main_task"]["AUC"], logs_final_validation["main_task"]["ACC"], logs_final_validation["main_task"]["PPV"],
        logs_final_validation["main_task"]["Sensitivity"], logs_final_validation["main_task"]["Fscore"], logs_final_validation["main_task"]["Sensitivity"]))

    print("-------------------------Testing Results--------------------------")
    print("{:>4s}\t{:>4s}\t{:>4s}\t{:>4s}\t{:>4s}\t{:>4s}".format(
        "AUC", "ACC", "PPV", "Sensitivity", "Fscore", "Sensitivity"))
    print("{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format(
        logs_final_test["main_task"]["AUC"], logs_final_test["main_task"]["ACC"], logs_final_test["main_task"]["PPV"],
        logs_final_test["main_task"]["Sensitivity"], logs_final_test["main_task"]["Fscore"], logs_final_test["main_task"]["Sensitivity"]))

if __name__ == '__main__':
    main()