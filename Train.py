import torch
import math
# from transformers import Token
import matplotlib.pyplot as plt
from tqdm import tqdm


class Train:
    def Train(model, train_DL, val_DL, criterion, optimizer, scheduler = None):
        loss_history = {"train": [], "val": []}
        best_loss = 9999
        for ep in range(EPOCH):
            model.train() # train mode로 전환
            train_loss = loss_epoch(model, train_DL, criterion, optimizer = optimizer, scheduler = scheduler)
            loss_history["train"] += [train_loss]

            model.eval() # test mode로 전환
            with torch.no_grad():
                val_loss = loss_epoch(model, val_DL, criterion)
                loss_history["val"] += [val_loss]
                if val_loss < best_loss:
                    best_loss = val_loss
                    torch.save({"model": model,
                                "ep": ep,
                                "optimizer": optimizer,
                                "scheduler": scheduler,}, save_model_path)
            # print loss
            print(f"Epoch {ep+1}: train loss: {train_loss:.5f}   val loss: {val_loss:.5f}   current_LR: {optimizer.param_groups[0]['lr']:.8f}")
            print("-" * 20)

        torch.save({"loss_history": loss_history,
                    "EPOCH": EPOCH,
                    "BATCH_SIZE": BATCH_SIZE}, save_history_path)

    def Test(model, test_DL, criterion):
        model.eval() # test mode로 전환
        with torch.no_grad():
            test_loss = loss_epoch(model, test_DL, criterion)
        print(f"Test loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):.3f}")

    def loss_epoch(model, DL, criterion, optimizer = None, scheduler = None):
        N = len(DL.dataset) # the number of data

        rloss=0
        for src_texts, trg_texts in tqdm(DL, leave=False):
            src = tokenizer(src_texts, padding=True, truncation=True, max_length = max_len, return_tensors='pt', add_special_tokens = False).input_ids.to(DEVICE)
            trg_texts = ['</s> ' + s for s in trg_texts]
            trg = tokenizer(trg_texts, padding=True, truncation=True, max_length = max_len, return_tensors='pt').input_ids.to(DEVICE)
            # inference
            y_hat = model(src, trg[:,:-1])[0] # 모델 통과 시킬 땐 trg의 마지막 토큰은 제외!
            # y_hat.shape = 개단차 즉, 훈련 땐 문장이 한번에 튀어나옴
            # loss
            loss = criterion(y_hat.permute(0,2,1), trg[:,1:]) # loss 계산 시엔 <sos> 는 제외!
            """
            개단차 -> 개차단으로 바꿔줌 (1D segmentation으로 생각)
            개채행열(예측), 개행열(정답)으로 주거나 개채1열, 개1열로 주거나 개채열, 개열로 줘야하도록 함수를 만들어놔서
            우리 상황에서는 개차단, 개단 으로 줘야 한다.
            이렇게 함수를 만들어놔야 1D, 2D segmentation 등등으로 확장가능하기 때문
            다 필요없고, 그냥 y_hat=개차단, trg=개단으로 줘야만 계산 제대로 된다고 생각하시면 됩니다!
            """
            # update
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            # loss accumulation
            loss_b = loss.item() * src.shape[0]
            rloss += loss_b
        loss_e = rloss/N
        return loss_e

    def count_params(model):
        num = sum([p.numel() for p in model.parameters() if p.requires_grad])
        return num

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

def plot_scheduler(scheduler_name, optimizer, scheduler, total_steps): # LR curve 보기
    lr_history = []
    steps = range(1, total_steps)

    for _ in steps: # base model -> 10만 steps (12시간), big model -> 30만 steps (3.5일) 로 훈련했다고 함
        lr_history += [optimizer.param_groups[0]['lr']]
        scheduler.step()

    plt.figure()
    if scheduler_name == 'Noam':
        if total_steps == 100000:
            plt.plot(steps, (512 ** -0.5) * torch.tensor(steps) ** -0.5, 'g--', linewidth=1, label=r"$d_{\mathrm{model}}^{-0.5} \cdot \mathrm{step}^{-0.5}$")
            plt.plot(steps, (512 ** -0.5) * torch.tensor(steps) * 4000 ** -1.5, 'r--', linewidth=1, label=r"$d_{\mathrm{model}}^{-0.5} \cdot \mathrm{step} \cdot \mathrm{warmup\_steps}^{-1.5}$")
        plt.plot(steps, lr_history, 'b', linewidth=2, alpha=0.8, label="Learning Rate")
    elif scheduler_name == 'Cos':
        plt.plot(steps, lr_history, 'b', linewidth=2, alpha=0.8, label="Learning Rate")
    plt.ylim([-0.1*max(lr_history), 1.2*max(lr_history)])
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.grid()
    plt.legend()
    plt.show()