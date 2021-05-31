echo Starting sampling
pause

call conda activate pytorch
python sample_grad_madry.py -output ..\samples -c ..\runs\mnist\pretrained -g ..\runs\mnist\dcgan-mnist -samples 1000 -ratio 0.3 --natural
python sample_grad_madry.py -output ..\samples -c ..\runs\mnist\pretrained -g ..\runs\mnist\dcgan-mnist -samples 1000 -ratio 0.3 --robust
echo Sampling complete
pause