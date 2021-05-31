echo Starting training
pause

call conda activate pytorch
::python train.py -t cls -m ../models/models/cls_mnist.py -d mnist -o ../runs/mnist --tag cls-mnist --batch_size 128 --epochs 20
python train.py -t cls -m ../models/models/cls_cifar.py -d cifar10 -o ../runs/cifar10 --tag cls-cifar --batch_size 128 --epochs 100
python train.py -t cls -m ../models/models/cls_svhn.py -d svhn -o ../runs/svhn --tag cls-svhn --batch_size 128 --epochs 50
echo Training complete
pause