# "conda create -n train python=3.7"
# "conda activate train"

# conda install python=3.7
# conda install pip
export INSTALL_DIR=$PWD
cat requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 pip install

# # apt-get -qq install -y python-prctl

git clone git@github.com:NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

cd /home/vilbert-multi-task
python setup.py develop

pip uninstall -y protobuf
pip install --no-binary=protobuf protobuf

pip install tensorboardX==1.8

cd tools/refer
python setup.py install
make

cd /home/vilbert-multi-task
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

#apex?

cd /home/vilbert-multi-task
git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git
cd vqa-maskrcnn-benchmark
python setup.py build develop

cd /home/vilbert-multi-task
pip install scipy==1.1.0
pip install pytorch-transformers==1.2.0

git clone https://github.com/Cadene/vqa-maskrcnn-benchmark.git
cd vqa-maskrcnn-benchmark
# # wget https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth

# # mv ./e2e_mask_rcnn_R_50_FPN_1x.pth /home/silly5921/vilbert-multi-task/save/resnext_models/R50FPN_model.pth

# wget https://dl.fbaipublicfiles.com/pythia/detectron_model/FAST_RCNN_MLP_DIM2048_FPN_DIM512.pkl


# Conda Environment
conda conda install -c pytorch torchvision
conda install -c conda-forge imageio
conda install -c anaconda nltk