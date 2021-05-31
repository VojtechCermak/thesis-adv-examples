echo Starting training
pause

call conda activate pytorch
::python train.py -t cls_at -m models/models/cls_mnist_oh.py -d "AUG-ALI-GRAD-N0_1-OR0_5" -o ../runs/mnist --tag "cls_AUG-ALI-GRAD-N0_1-OR0_5" --batch_size 256 --epochs 500
::python train.py -t cls_at -m models/models/cls_mnist_oh.py -d "AUG-ALI-GRAD-N0_0-OR0_5" -o ../runs/mnist --tag "cls_AUG-ALI-GRAD-N0_0-OR0_5" --batch_size 256 --epochs 500

python train.py -t cls_at -m models/models/cls_mnist_oh.py -d "AUG-ALI-GRAD-N0_0-OR0_95" -o ../runs/mnist --tag "cls_AUG-ALI-GRAD-N0_0-OR0_95-sched" --batch_size 256 --epochs 200
::python train.py -t cls_at -m models/models/cls_mnist_oh.py -d "AUG-ALI-GRAD-N0_0-OR0_05" -o ../runs/mnist --tag "cls_AUG-ALI-GRAD-N0_0-OR0_05"

echo Training complete
pause