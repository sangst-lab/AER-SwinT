import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn import metrics

class Epoch:

    def __init__(self, model, loss, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss = loss
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        #for metric in self.metrics:
        #    metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):
        self.on_epoch_start()

        logs = {}
        loss_meter_main = AverageValueMeter()
        loss_meter_sub1 = AverageValueMeter()
        loss_meter_sub2 = AverageValueMeter()
        ##
        #
        # ##
        target_main_list = []
        target_sub1_list = []
        target_sub2_list = []
        pred_main_list=[]
        predlabel_main_list=[]
        pred_sub1_list=[]
        predlabel_sub1_list=[]
        pred_sub2_list=[]
        predlabel_sub2_list=[]

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for organ_HU, mask_gastcar, tabular_data, alllabels, RoI in iterator:
                #alllabels=[(main_task, sub_task_1, sub_task_2)...]
                #x, y = organ_HU.to(self.device), alllabels[0].type(torch.float32).to(self.device)
                #x, y = organ_HU.to(self.device), alllabels[0].type(torch.float32).to(self.device)
                #img, tabdata, labels = organ_HU.to(self.device), tabular_data.type(torch.float32).to(self.device), alllabels
                img, tabdata, labels = RoI.to(self.device), tabular_data.type(torch.float32).to(
                    self.device), alllabels
                target_main=alllabels[0].type(torch.float32).to(self.device)
                target_sub1 = alllabels[1].type(torch.float32).to(self.device)
                target_sub2 = alllabels[2].type(torch.float32).to(self.device)

                loss_main, loss_sub1, loss_sub2, pred_main, pred_sub1, pred_sub2 = self.batch_update(img, tabdata, target_main, target_sub1, target_sub2)

                # save the target and pred for calculating AUC
                # For main task
                self.save_prediction(target_main, pred_main, target_main_list, pred_main_list, predlabel_main_list)
                self.save_prediction(target_sub1, pred_sub1, target_sub1_list, pred_sub1_list, predlabel_sub1_list)
                self.save_prediction(target_sub2, pred_sub2, target_sub2_list, pred_sub2_list, predlabel_sub2_list)

                # update loss logs
                loss_meter_main.add(loss_main.cpu().detach().numpy())
                loss_meter_sub1.add(loss_sub1.cpu().detach().numpy())
                loss_meter_sub2.add(loss_sub2.cpu().detach().numpy())

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        logs["main_task"] = self.performance(target_main_list, pred_main_list, predlabel_main_list)
        logs["sub1_task"] = self.performance(target_sub1_list, pred_sub1_list, predlabel_sub1_list)
        logs["sub2_task"] = self.performance(target_sub2_list, pred_sub2_list, predlabel_sub2_list)
        logs["main_task"]["Loss"] = loss_meter_main.mean
        logs["sub1_task"]["Loss"] = loss_meter_sub1.mean
        logs["sub2_task"]["Loss"] = loss_meter_sub2.mean
        return logs

    def performance(self,targets, probs, probs_label):
        # 计算 AUC
        fpr, tpr, _ = roc_curve(targets, probs)
        AUC = auc(fpr, tpr)

        # 计算 ACC
        ACC = metrics.accuracy_score(targets, probs_label)

        # 计算 PPV (precision)
        PPV = metrics.precision_score(targets, probs_label)

        # 计算 Sensitivity (recall)
        Sensitivity = metrics.recall_score(targets, probs_label)

        # 计算 F-score
        Fscore = metrics.f1_score(targets, probs_label)

        # 计算 Specificity
        tn, fp, fn, tp = metrics.confusion_matrix(targets, probs_label).ravel()
        #print(tp, fp, tn, fn)
        Specificity = tn / (tn + fp)

        # 计算 NPV
        #NPV = tn / (fn + tn)

        return {"AUC":AUC, "ACC":ACC, "PPV":PPV, "Sensitivity":Sensitivity,
        "Fscore":Fscore,
        #logs["NPV"]=NPV,
        "Specificity":Specificity,
        "Probs":probs,
        "Probs_label":probs_label,
        "Targets":targets}

    def save_prediction(self, target, pred, target_list, pred_list, prelabel_list):
        # For main task
        for i in target:
            target_list.append(i.item())

        #for i in pred[:, 0]:
        for i in pred:
            pred_list.append(i.item())

        #for i in pred[:, 0]:
        for i in pred:
            if i.item() >= 0.5:
                prelabel_list.append(1.0)
            else:
                prelabel_list.append(0.0)

class TrainEpoch(Epoch):

    def __init__(self, model, loss, optimizer, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, img, tabdata, target_main, target_sub1, target_sub2):
        self.optimizer.zero_grad()

        prediction_main, prediction_sub1, prediction_sub2 = self.model.forward(img, tabdata)

        #loss, loss_sub1, loss_sub2 = self.loss(prediction_main[:, 0], prediction_sub1[:, 0], prediction_sub2[:, 0], target_main, target_sub1, target_sub2)
        loss, loss_sub1, loss_sub2 = self.loss(prediction_main, prediction_sub1, prediction_sub2,
                                               target_main, target_sub1, target_sub2)
        loss.backward()
        self.optimizer.step()
        return loss, loss_sub1, loss_sub2, prediction_main, prediction_sub1, prediction_sub2


class ValidEpoch(Epoch):

    def __init__(self, model, loss, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, img, tabdata, target_main, target_sub1, target_sub2):
        with torch.no_grad():
            prediction_main, prediction_sub1, prediction_sub2 = self.model.forward(img, tabdata)

            loss, loss_sub1, loss_sub2 = self.loss(prediction_main, prediction_sub1, prediction_sub2,
                                                   target_main, target_sub1, target_sub2)

        return loss, loss_sub1, loss_sub2, prediction_main, prediction_sub1, prediction_sub2
