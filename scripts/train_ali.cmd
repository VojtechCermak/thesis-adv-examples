echo Starting training
pause

call conda activate pytorch
python train.py -t end -m ../models/models/ali_mnist.py -d mnist -o ../runs/mnist --tag ali-mnist --batch_size 64 --epochs 50
python train.py -t end -m ../models/models/ali_cifar.py -d cifar10 -o ../runs/cifar10 --tag ali-cifar --batch_size 128 --epochs 500
python train.py -t end -m ../models/models/ali_svhn.py -d svhn -o ../runs/svhn --tag ali-svhn --batch_size 128 --epochs 200
echo Training complete
pause