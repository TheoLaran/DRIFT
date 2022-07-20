NB_E=1800
python3.7 write.py &
nohup python3.7 Launcher.py 1 $NB_E 0 > nohup/"saros.txt" &
nohup python3.7 Launcher.py 2 $NB_E 0 > nohup/"out_0.txt" &


