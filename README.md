# DRIFT

To launch the code you need to add your dataset in the folder DRIFT/dataset, and its preprocessing in DRIFT/Preprocess.py 

To Launch the training use python3 Launcher.py with options it can be : <br>
                  <li> -b to set the minimum number of blocks update.
                  <li> -e to set the minimum number of epoch.
                  <li> -t to set the minimum time before ending the update.
                  <li> -g to set the target genre.
                  <li> -d to set a dataset (MovieLens by default).
                  <li> -p to predict the recommendation (False by default).
                  <li> -w to write the JSON result (False by default).
                  

To plot the JSON results use python3 DRIFT/utils.py f, with f the name of the JSON you want to plot. 

The theoritical part of this code is available in the report. 
