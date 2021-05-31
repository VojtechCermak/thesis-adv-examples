echo Starting training
pause

call conda activate pytorch
python train.py -t end -m ../models/models/vae_mnist.py -d mnist -o ../runs/mnist --tag vae-mnist --batch_size 128 --epochs 80
python train.py -t end -m ../models/models/vae_cifar.py -d cifar10 -o ../runs/cifar10 --tag vae-cifar --batch_size 128 --epochs 150
python train.py -t end -m ../models/models/vae_svhn.py -d svhn -o ../runs/svhn --tag vae-svhn --batch_size 128 --epochs 100
echo Training complete
pause