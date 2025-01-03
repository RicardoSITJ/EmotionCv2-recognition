```
conda create --name myenv python=3.9

conda activate myenv 
```

```
pip install Pillow  
pip install opencv-python 
pip install imutils     
pip install keras
pip install tensorflow  
pip install scipy      
pip install pandas  
pip install matplotlib  
pip install imageio
```

```
conda env export --no-builds > environment.yml

conda deactivate

conda remove --name <environment_name> --all

conda env list

conda env create -f environment.yml

conda activate myenv
```

```
python DETECTOR.py
```