对于论文DeepFaceDrawing: Deep Generation of Face Images from Sketches的复现



使用环境为

+ python3.7
+ pytorch

# 0 Directory Structure

for datasets

```
DeepFaceDrawing
└───datasets
    └───train
    |	 └───Image
    |    |    | <file0>.jpg
    |    |    | <file1>.jpg
    |    |    
    |    └───Edge
    |         | <file0>.jpg
    |         | <file1>.jpg
    |   
    └───test
         └───Image
         |    | <file0>.jpg
         |    | <file1>.jpg
         |    
         └───Edge
              | <file0>.jpg
              | <file1>.jpg
```





# 1 AE

run the following code to train auto encoder

```
python auto_encoder_train.py
```



# 2 Image Generator

run the following code to train image generator

```
python image_generator_train.py
```

