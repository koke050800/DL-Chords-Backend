import os
import requests

# server url
URL = "http://localhost:4000/predict"
URL_PROD = "http://3.145.204.250/predict"
URL_DEV = "http://127.0.0.1/predict"


# audio file we'd like to send for predicting keyword
FILE_PATH = "C:/Users/Omar/Documents/DeepLearning/dlchordsserver/dlchordsserver/cuando-me-enamoro.mp3"


if __name__ == "__main__":

    # open files
   # cwd = os.getcwd()  # Get the current working directory (cwd)
    #files = os.listdir(cwd)  # Get all the files in that directory
    #print("Files in %r: %s" % (cwd, files))
    file = open(FILE_PATH, "rb")
    print(file)
    # package stuff to send and perform POST request
    values = {"file": (FILE_PATH, file) }
    pruebas = {"time_initial": "40", "time_final": "100"}
    response = requests.post(URL, files=values, headers=pruebas)
    data = response.json()

    print("Todo lo devuelto: {}".format(data))