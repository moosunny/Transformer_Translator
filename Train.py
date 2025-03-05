import torch
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

class NoamScheduler:
    def __init__(self, optimizer, d_model, warmup_steps, LR_scale = 1):
        self.optimizer = optimizer
        self.current_step = 0
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.LR_scale = LR_scale

    def step(self):
        self.current_step += 1
        lrate = self.LR_scale * (self.d_model ** -0.5) * min(self.current_step ** -0.5, self.current_step * self.warmup_steps ** -1.5)
        self.optimizer.param_groups[0]['lr'] = lrate


class Trainer:
    def __init__(self, model, train_DL, val_DL, epoch, BATCH_SIZE, DEVICE, max_len, tokenizer, scheduler, criterion, optimizer, save_model_path, save_history_path):
        self. model = model
        self.train_DL = train_DL
        self.val_DL = val_DL
        self.epoch = epoch
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.DEVICE = DEVICE
        self.BATCH_SIZE = BATCH_SIZE
        self.tokenizer = tokenizer  
        self.max_len = max_len
        self.save_model_path = save_model_path
        self.save_history_path = save_history_path

    def loss_epoch(self, DL, train = True):
        N = len(DL.dataset) # the number of data
        rloss=0
        for src_texts, trg_texts in tqdm(DL, leave=False):
            src = self.tokenizer(src_texts, padding=True, truncation=True, max_length = self.max_len, return_tensors='pt', add_special_tokens = False)["input_ids"].to(self.DEVICE)
            # print(src)
            trg_texts = ['</s> ' + s for s in trg_texts]
            trg = self.tokenizer(trg_texts, padding=True, truncation=True, max_length = self.max_len, return_tensors='pt')["input_ids"].to(self.DEVICE)
            # inference
            y_hat = self.model(src, trg[:,:-1])[0] # 모델 통과 시킬 땐 trg의 마지막 토큰은 제외!
            loss = self.criterion(y_hat.permute(0,2,1), trg[:,1:]) # loss 계산 시엔 <sos> 는 제외!

            # update
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()
            # loss accumulation
            loss_b = loss.item() * src.shape[0]
            rloss += loss_b
        loss_e = rloss/N
        return loss_e



    def count_params(self):
        num = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        return num

    def Train(self):
        loss_history = {"train": [], "val": []}
        best_loss = 9999
        for ep in range(self.epoch):
            self.model.train() # train mode로 전환
            train_loss = self.loss_epoch(DL = self.train_DL)
            loss_history["train"] += [train_loss]

            self.model.eval() # test mode로 전환
            with torch.no_grad():
                val_loss = self.loss_epoch(DL = self.val_DL, train = False)
                loss_history["val"] += [val_loss]
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save({"model": self.model,
                                "ep": ep,
                                "optimizer": self.optimizer,
                                "scheduler": self.scheduler,}, self.save_model_path)
            # print loss
            print(f"Epoch {ep+1}: train loss: {train_loss:.5f}   val loss: {val_loss:.5f}   current_LR: {self.optimizer.param_groups[0]['lr']:.8f}")
            print("-" * 20)

        torch.save({"loss_history": loss_history,
                    "EPOCH": self.epoch,
                    "BATCH_SIZE": self.BATCH_SIZE}, self.save_history_path)
        
    def Test(self, test_DL):
        self.model.eval()  # test mode로 전환
        with torch.no_grad():
            test_loss = self.loss_epoch(test_DL, train = False)
        print(f"Test loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}")
        

    def plot_scheduler(self, total_steps): # LR curve 보기
        lr_history = []
        steps = range(1, total_steps)

        for _ in steps: # base model -> 10만 steps (12시간), big model -> 30만 steps (3.5일) 로 훈련했다고 함
            lr_history += [self.optimizer.param_groups[0]['lr']]
            self.scheduler.step()

        plt.figure()
        if self.scheduler_name == 'Noam':
            if total_steps == 100000:
                plt.plot(steps, (512 ** -0.5) * torch.tensor(steps) ** -0.5, 'g--', linewidth=1, label=r"$d_{\mathrm{model}}^{-0.5} \cdot \mathrm{step}^{-0.5}$")
                plt.plot(steps, (512 ** -0.5) * torch.tensor(steps) * 4000 ** -1.5, 'r--', linewidth=1, label=r"$d_{\mathrm{model}}^{-0.5} \cdot \mathrm{step} \cdot \mathrm{warmup\_steps}^{-1.5}$")
            plt.plot(steps, lr_history, 'b', linewidth=2, alpha=0.8, label="Learning Rate")
        elif self.scheduler_name == 'Cos':
            plt.plot(steps, lr_history, 'b', linewidth=2, alpha=0.8, label="Learning Rate")
        plt.ylim([-0.1*max(lr_history), 1.2*max(lr_history)])
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.grid()
        plt.legend()
        plt.show()
