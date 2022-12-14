# INF634 Adding and removing masks to/from people using GANs 

## Setup
1. Downloading and preparing data.
   1. Download data from [here](https://www.kaggle.com/datasets/akashguna/lfw-dataset-with-masks)
   2. Unpack it and move it to the root of the project
   3. Rename "archive" folder into "data" folder
   4. Create "data_transformed" folder
   5. Create "mask_to_no_mask", "no_mask_to_mask" and "cycle_transform" folders inside "data_transformed" folder
   6. With pretrained Cycle GAN create normal-LFW and put it in the "mask_to_no_mask" folder
   7. With pretrained Cycle GAN create masked-LFW and put it in the "no_mask_to_mask" folder
   8. With pretrained Cycle GAN create cycle-LFW and put it in the "cycle_transform" folder
   Note the structure should be "data_transformed/A/images/*_fake.png" where A in {"cycle_transform", "mask_to_no_mask", "no_mask_to_mask"}
   
   ** normal-LFW, masked-LFW, cycle-LFW can be also downloaded [here](https://drive.google.com/drive/folders/1KWRvolS6zHGbqJmcDuXM3HJP5a7x2IIo?usp=sharing)
2. Installing required modules
    ```
    pip install -r requirements.txt
    ```

## Launch
Test checkpoint:
1. Name it last.ckpt
2. Put it into checkpoints folder
3. Run test script
   ```
   python -m src.train_test_identification
   ```

Train & test model:
```
python -m src.train_test_identification --train
```

## Train, test and evaluate CycleGAN
1. Clone the repository
2. Run
```
mv INF634/* .
```
3. Download data (achieve.zip) from [here](https://www.kaggle.com/datasets/akashguna/lfw-dataset-with-masks)
4. Put achieve.zip in the same directory where CycleGAN.ipynb resides
3. Run CycleGAN.ipynb to train, test and evaluate CycleGAN (once you ran the notebook, you will get normal-LFW, masked-LFW, cycle-LFW datasets in the folder pytorch-CycleGAN-and-pix2pix).
