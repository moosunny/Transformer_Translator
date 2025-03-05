import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from Transformer import *
from transformers import MarianMTModel, MarianTokenizer # MT: Machine Translation
from Train import *
from Set_Seed import SetSeed
from Datasets import CustomDataset
from BLEU_Score import *

SetSeed.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ko-en')
data = pd.read_excel('2_대화체.xlsx')

BATCH_SIZE = 64 # 논문에선 2.5만 token이 한 batch에 담기게 했다고 함.
LAMBDA = 0 # l2-Regularization를 위한 hyperparam. # 저장된 모델
EPOCH = 3 # 저장된 모델

eos_idx = tokenizer.eos_token_id
pad_idx = tokenizer.pad_token_id
vocab_size = tokenizer.vocab_size


# max_len = 512 # model.model.encoder.embed_positions 를 보면 512로 했음을 알 수 있다.
max_len = 100 # 너무 긴거 같아서 자름 (GPU 부담도 많이 덜어짐)
criterion = nn.CrossEntropyLoss(ignore_index = pad_idx) # pad token 이 출력 나와야하는 시점의 loss는 무시 (즉, label이 <pad> 일 때는 무시) # 저장된 모델
# criterion = nn.CrossEntropyLoss(ignore_index = pad_idx, label_smoothing = 0.1) # 막상 해보니 성능 안나옴 <- 데이터가 많아야 할 듯

scheduler_name = 'Noam'
# scheduler_name = 'Cos'
#### Noam ####
# warmup_steps = 4000 # 이건 논문에서 제시한 값 (총 10만 step의 4%)
warmup_steps = 1000 # 데이터 수 * EPOCH / BS = 총 step 수 인것 고려 # 저장된 모델
LR_scale = 0.5 # Noam scheduler에 peak LR 값 조절을 위해 곱해질 녀석 # 저장된 모델
#### Cos ####
LR_init = 5e-4
T0 = 1500 # 첫 주기
T_mult = 2 # 배 만큼 주기가 길어짐 (1보다 큰 정수여야 함)
#############

save_model_path = '/Transformer_EPOCH=15.pt'
save_history_path = '/Transformer_EPOCH=15_history.pt'

# 좀 사이즈 줄인 모델 (훈련된 input_embedding, fc_out 사용하면 사용 불가)
n_layers = 3
d_model = 256
d_ff = 512
n_heads = 8
drop_p = 0.1

custom_DS = CustomDataset(data)

train_DS, val_DS, test_DS = torch.utils.data.random_split(custom_DS, [97000, 2000, 1000])
# 논문에서는 450만개 영,독 문장 pair 사용

train_DL = torch.utils.data.DataLoader(train_DS, batch_size=BATCH_SIZE, shuffle=True)
val_DL = torch.utils.data.DataLoader(val_DS, batch_size=BATCH_SIZE, shuffle=True)
test_DL = torch.utils.data.DataLoader(test_DS, batch_size=BATCH_SIZE, shuffle=True)

print(len(train_DS))
print(len(val_DS))
print(len(test_DS))

model = Transformer(vocab_size, max_len, n_layers, d_model, d_ff, n_heads, drop_p, pad_idx, DEVICE).to(DEVICE)

optimizer = optim.Adam(nn.Linear(1, 1).parameters(), lr=0)
# scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps, LR_scale=LR_scale)
scheduler_name = "Noam"

params = [p for p in model.parameters() if p.requires_grad] # 사전 학습된 layer를 사용할 경우
if scheduler_name == 'Noam':
    optimizer = optim.Adam(params, lr=0,
                        betas=(0.9, 0.98), eps=1e-9,
                        weight_decay=LAMBDA) # 논문에서 제시한 beta와 eps 사용, l2-Regularization은 한번 써봄 & 맨 처음 step 의 LR=0으로 출발 (warm-up)
    scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps, LR_scale=LR_scale)

elif scheduler_name == 'Cos':
    optimizer = optim.Adam(params, lr=LR_init,  # cos restart sheduling
                        betas=(0.9, 0.98), eps=1e-9,
                        weight_decay=LAMBDA)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T0, T_mult)

trainer = Trainer(model = model, train_DL = train_DL, val_DL = val_DL, epoch = EPOCH, BATCH_SIZE = BATCH_SIZE, DEVICE = DEVICE, max_len = max_len, tokenizer = tokenizer, scheduler=scheduler ,criterion = criterion, optimizer = optimizer, save_model_path = save_model_path, save_history_path = save_history_path)
print("파라미터 수:",trainer.count_params(), "개")
trainer.Train()


loaded = torch.load(save_model_path, map_location=DEVICE)
load_model = loaded["model"]
ep = loaded["ep"]
optimizer = loaded["optimizer"]

loaded = torch.load(save_history_path, map_location=DEVICE)
loss_history = loaded["loss_history"]

print(ep)
print(optimizer)

plt.figure()
plt.plot(range(1,EPOCH+1),loss_history["train"], label="train")
plt.plot(range(1,EPOCH+1),loss_history["val"], label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train, Val Loss")
plt.grid()
plt.legend()

trainer.Test(test_DL)

# Perplxity 구하기
y_hat = torch.tensor([[[0.3659, 0.7025, 0.3104]], [[0.0097, 0.6577, 0.1947]],[[0.5659, 0.0025, 0.0104]], [[0.9097, 0.0577, 0.7947]]])
target = torch.tensor([[2],  [1], [2],  [1]])

soft = nn.Softmax(dim=-1)
y_hat_soft = soft(y_hat)
print(y_hat_soft.shape)
v=1
for i, val in enumerate(y_hat_soft):
    v*=val[0,target[i]]
print(v**(-1/target.shape[0]))
# 3.5257

criterion_test = nn.CrossEntropyLoss()
print(y_hat.permute(0,2,1).shape)
print(target.shape)
print(torch.exp(criterion_test(y_hat.permute(0,2,1), target))) # 결론: loss에 torch.exp 취하셈
# 3.5257