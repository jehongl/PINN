# PINN

```
conda create --name PINN_fluid_model python=3.9  
source activate PINN_fluid_model
```

```
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install opencv-python
```

to-do list

1. build the pinn model
2. training in simple domain
3. read the paper about neural operator