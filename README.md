# NER_yz

1.Dataset

采用 BMEO 标注体系，数据格式:

```
新 B-ORG
华 M-ORG
社 E-ORG
上 B-GPE
海 E-GPE
二 O
月 O
十 O
日 O
电 O
（ O
记 O
者 O
谢 B-PER
金 M-PER
虎 E-PER
、 O
张 B-PER
持 M-PER
坚 E-PER
） O
```

2.Train

```
python main.py --do_train --num_train_epochs 30
```
