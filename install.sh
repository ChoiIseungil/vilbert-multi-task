# cat requirements.txt | sed -e '/^\s*#.*$/d' -e '/^\s*$/d' | xargs -n 1 pip install

# # apt-get -qq install -y python-prctl

# # git clone https://github.com/NVIDIA/apex

# # cd apex
# # pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
# # cd ../

python setup.py develop

# # pip uninstall -y protobuf
# # pip install --no-binary=protobuf protobuf

# # pip install tensorboardX==1.8

# # # rm -r tools/refer
# # # mkdir tools/refer
# # # git clone -b python3 https://github.com/lichengunc/refer tools/refer

cd tools/refer
python setup.py install
make
cd ..
cd ..

# # # git clone https://github.com/cocodataset/cocoapi.git

cd cocoapi/PythonAPI
python setup.py build_ext install
cd ..
cd ..

# # git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts
python setup.py build_ext install
cd .. 

# # git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git
# cd vqa-maskrcnn-benchmark
# python setup.py build develop
# cd ../

# # pip install scipy==1.1.0
# # pip install pytorch-transformers==1.2.0

# # wget https://dl.fbaipublicfiles.com/vilbert-multi-task/pretrained_model.bin

# # cd vqa-maskrcnn-benchmark
# # wget https://download.pytorch.org/models/maskrcnn/e2e_mask_rcnn_R_50_FPN_1x.pth

# # mv ./e2e_mask_rcnn_R_50_FPN_1x.pth /home/silly5921/vilbert-multi-task/save/resnext_models/R50FPN_model.pth

# wget https://dl.fbaipublicfiles.com/pythia/detectron_model/FAST_RCNN_MLP_DIM2048_FPN_DIM512.pkl

