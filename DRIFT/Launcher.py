from DRIFT.main import main
import sys
from utils.utils import GENRES

def launcher(words):
    args = {
        "MIN_b": 0,
        "MIN_t": 0,
        "MIN_e": 0,
        "target_genre": "Main",
        "dataset" : "ML",
        "write": False,
        "predict": False,
    }
    for i in range(0, len(words), 2):
        command = words[i]
        if command == "-b":
            args["MIN_b"] = int(words[i + 1])
        elif command == "-e":
            args["MIN_e"] = int(words[i + 1])
        elif command == "-t":
            args["MIN_t"] = int(words[i + 1])
        elif command == "-d":
            args["dataset"] = words[i + 1]
        elif command == "-w":
            args["write"] = bool(int(words[i + 1]))
        elif command == "-p":
            args["predict"] = bool(int(words[i + 1]))
        elif command == "-g":
            t_g = int(words[i + 1])
            if t_g == -1 :
                genre = "SAROS"
            else :
                genre = GENRES[t_g]
            args["target_genre"] = genre
        elif command == "-h":
            print(f"\n\n usage python3.7 Launcher.py opt \n with opt in : \n"
                  f"\t * -b to set the minimum number of blocks updated\n"
                  f"\t * -e to set the minimum number of epoch\n"
                  f"\t * -t to set the minimum time before ending the update\n"
                  f"\t * -g to set the target genre (-1 to launch SAROS)\n"
                  f"\t * -d to set a dataset (ML by default)\n"
                  f"\t * -p to predict the recommendation (False by default)\n"
                  f"\t * -w to write the JSON result into (False by default)\n"
                  f"\t * -h to display this menu\n"
                  )
            return
        else:
            raise Exception(f"Unknown command {command}, use python3.7 Launcher.py -h to get the help")
    return main(args)

launcher(sys.argv[1:])



