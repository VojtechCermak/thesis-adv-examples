echo Starting training
pause

call conda activate pytorch
python train.py -t gen -m ../models/models/dcgan_mnist.py -d mnist -o ../runs/mnist --tag dcgan-mnist --batch_size 64 --epochs 100
python train.py -t gen -m ../models/models/dcgan_cifar.py -d cifar10 -o ../runs/cifar10 --tag dcgan-cifar --batch_size 64 --epochs 200
python train.py -t gen -m ../models/models/dcgan_svhn.py -d svhn -o ../runs/svhn --tag dcgan-svhn --batch_size 64 --epochs 150
echo Training complete
pause