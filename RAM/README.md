# Deployment details for Risk Assessment Module
```bash=
conda create --name crash python=3.9
conda activate crash

pip install "pip<24.1"
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c conda-forge ffmpeg av

# Build detectron2 from source
pip install ninja
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
python -m pip install . --no-build-isolation

# Clean the old build if needed
rm -rf build/ **/*.so

# Install the remaining requirements
pip install -r requirements.txt
pip install -U openmim
mim install mmcv==2.2.0
mim install mmdet==3.3.0
```