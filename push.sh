# git config --global user.email "silly5921@kaist.ac.kr"
# git config --global user.name "ChoiIseungil"

git add -u
#큰 파일은 nas2에 넣기
# git reset -- master/pretrained_model.bin master/features/FGA50000_extracted_features master/FAST_RCNN_MLP_DIM2048_FPN_DIM512.pkl  master/FAST_RCNN_MLP_DIM2048_FPN_DIM512.pkl.1 master/FAST_RCNN_MLP_DIM2048_FPN_DIM512.pkl master/data master/vqa-maskrcnn-benchmark/FAST_RCNN_MLP_DIM2048_FPN_DIM512.pkl master/dataset/
git commit -m "json generator started"

# git push https://ChoiIseungil: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@github.com/ChoiIseungil/vilbert-multi-task.git -u origin master --recurse-submodules=on-demand
git push -u origin master
# git push -u origin jsongenerator
