# 論文題目

## Setting up datasets
* Download BDDA dataset following the instructions on [official websites](http://bdd-data.berkeley.edu/download.html).
* Download DR(eye)VE dataset following the instructions on [official websites](https://aimagelab-legacy.ing.unimore.it/imagelab/page.asp?IdPage=8).
* Download TrafficGaze dataset following the instructions on [huggingface](https://huggingface.co/datasets/springyu/TrafficGaze).
* Download MM-AU dataset following the instructions on [huggingface](https://huggingface.co/datasets/JeffreyChou/MM-AU/tree/main).

---
## Driver Attention Module
Modeified from：https://github.com/ykotseruba/SCOUT

You can find instructions on how to set up the environment [here]().

### Training the model
Once you have completed the Deployment Details and made the necessary changes to [/config/DAM.yaml](), you can run the following command for training:
```bash
python3 train.py
```

### Testing the model
Place the config file and model weights in /your_config, then run the following command for:
```bash
python3 test.py --config_dir /your_config --evaluate
```

If you need to save test images, run the following command:
```bash
python3 test.py --config_dir /your_config --evaluate --save_images
```
---
## Risk Assessment Module
Modified from：https://github.com/DeSinister/CycleCrash/

You can find instructions on how to set up the environment [here]().

---
## LLM Inference Module
You can find instructions on how to set up the environment [here]().