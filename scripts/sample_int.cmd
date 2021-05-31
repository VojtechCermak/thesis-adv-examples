echo Starting sampling
pause

call conda activate pytorch
python sample_int.py -output ..\samples -c ..\runs\cifar10\cls-cifar -g ..\runs\cifar10\ali-cifar -samples 1000 -ratio 0.3 -sampler_batch_size 64 -sampler_threshold 0.95 -sampler_max_steps 2000
python sample_int.py -output ..\samples -c ..\runs\cifar10\cls-cifar -g ..\runs\cifar10\dcgan-cifar -samples 1000 -ratio 0.3 -sampler_batch_size 64 -sampler_threshold 0.95 -sampler_max_steps 2000

python sample_int.py -output ..\samples -c ..\runs\mnist\cls-mnist -g ..\runs\mnist\ali-mnist -samples 1000 -ratio 0.3 -sampler_max_steps 2000
python sample_int.py -output ..\samples -c ..\runs\mnist\cls-mnist -g ..\runs\mnist\dcgan-mnist -samples 1000 -ratio 0.3 -sampler_max_steps 2000
python sample_int.py -output ..\samples -c ..\runs\mnist\cls-mnist -g ..\runs\mnist\vae-mnist -samples 1000 -ratio 0.3 -sampler_max_steps 2000

python sample_int.py -output ..\samples -c ..\runs\svhn\cls-svhn -g ..\runs\svhn\ali-svhn -samples 1000 -ratio 0.3 -sampler_max_steps 2000
python sample_int.py -output ..\samples -c ..\runs\svhn\cls-svhn -g ..\runs\svhn\dcgan-svhn -samples 1000 -ratio 0.3 -sampler_max_steps 2000
python sample_int.py -output ..\samples -c ..\runs\svhn\cls-svhn -g ..\runs\svhn\vae-svhn -samples 1000 -ratio 0.3 -sampler_max_steps 2000
echo Sampling complete
pause