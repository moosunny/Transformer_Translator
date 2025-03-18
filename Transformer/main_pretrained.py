import torch
from transformers import MarianMTModel, MarianTokenizer # MT: Machine Translation
import pandas as pd
from Set_Seed import SetSeed
SetSeed.seed(42)

max_len = 100

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(DEVICE)

tokenizer = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-ko-en')
model = MarianMTModel.from_pretrained('Helsinki-NLP/opus-mt-ko-en')


print(" tokenizer 써보기 (_로 띄어쓰기를 나타낸다! 즉, _가 없으면 이어진 한 단어임을 나타냄 subword tokenizing)")
print(tokenizer.tokenize("Hi, I'm Hyuk. ...        a   a?"))
print(tokenizer.tokenize("a/b 1+2+3 2:1 a>b"))
print(tokenizer.tokenzize("pretrained restart"))
print(tokenizer.tokenize("chatGPT"))
print(tokenizer.tokenize("The example is very good in our lecture")) # 띄어쓰기도 tokenize 할 때가 있다.
print(tokenizer.tokenize("한글은 어떻게 할까?"))
print(tokenizer.tokenize("확실히 띄어쓰기 기준으로 토크나이징을 하는 것 같진 않다."))
print(tokenizer.tokenize("여러분들 차례!"))
print("="*50)

# print(tokenizer.get_vocab())
vocab_size = tokenizer.vocab_size
print(vocab_size)

print(tokenizer.encode('지능', add_special_tokens=False)) # string to index
print(tokenizer.encode('<pad>', add_special_tokens=False)) # <pad>는 65000
print(tokenizer.encode('</s>', add_special_tokens=False)) # <sos> or <eos>는 0
print(tokenizer.encode('He', add_special_tokens=False)) # add_special_tokens=False 는 <eos> 자동 붙여주는 것을 방지
print(tokenizer.encode('he', add_special_tokens=False)) # 대소문자 다른 단어로 인식
print(tokenizer.tokenize('문장을 넣으면 토크나이즈해서 숫자로 바꾼다'))
print(tokenizer.encode('문장을 넣으면 토크나이즈해서 숫자로 바꾼다', add_special_tokens=False))
# print(tokenizer.decode([204]))
# print(tokenizer.decode([206]))
# print(tokenizer.decode([210]))
# print(tokenizer.decode(list(range(15)) + [65000,65001,65002,65003]))

print("사전 학습된 모델로 번역해보기 (생각보다 성능 좋네)")
input_text = "헐! 대박 쩐다!"

input_tokens = tokenizer.encode(input_text, return_tensors="pt")
translated_tokens = model.generate(input_tokens, max_new_tokens=max_len)
translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

print("입력:", input_text)
print("AI의 번역:", translated_text)
