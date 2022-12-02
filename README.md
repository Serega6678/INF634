# INF634 Adding and removing masks to/from people using GANs 

## Setup
1. Downloading and preparing data.
   1. Download data from [here](https://www.kaggle.com/datasets/akashguna/lfw-dataset-with-masks)
   2. Unpack it and move it to the root of the project
   3. Rename "archive" folder into "data" folder
   4. Create "data_transformed" folder
   5. Create "mask_to_no_mask" and "no_mask_to_mask" folders inside "data_transformed" folder
   6. With pretrained Cycle GAN create normal-LFW and put it in the "mask_to_no_mask" folder
   7. With pretrained Cycle GAN create masked-LFW and put it in the "no_mask_to_mask" folder
   Note the structure should be "data_transformed/A/images/*_fake.png" where A in {"mask_to_no_mask", "no_mask_to_mask"}
2. Installing required modules
    ```
    pip install -r requirements.txt
    ```

## Launch
Start training and testing (including GAN-generated photos testing)
```
python -m src.train_identification
```
