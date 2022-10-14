import json
import os
import shutil
import random
from flask import Flask, request, jsonify
from functions.chord_spotting_service import Chord_Spotting_Service

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        return obj.__dict__  

app = Flask(__name__)

@app.route("/")
def home():
    return "La página está funcionando bien"

@app.route("/predict", methods=["POST", "GET"])
def predecir():
    audio_file = request.files["file"]
    print(audio_file)

    file_name = str(random.randint(0, 100000))
    audio_file.save(file_name)
    print(file_name)
	# instantiate chord spotting service singleton and get prediction
    chss = Chord_Spotting_Service()
    try:
        words_objects, chords_objects = chss.predict_song(file_name)
    except Exception as e:
        print("Eliminando archivos creados por exception..."+ e)
        shutil.rmtree("separate")
        os.remove(file_name)
        return {"Error": "Error en la predicción"}
    else:
        print("Codificando json...")
        
        words_json = json.dumps(words_objects, cls=MyEncoder, ensure_ascii=False)
        print(words_json) 
        chords_json = json.dumps(chords_objects, cls=MyEncoder)
        shutil.rmtree("separate")
        os.remove(file_name)
        result = {"words": words_json, "chords": chords_json}
        return jsonify(result)
    finally:
        print("Prediction is done...")
    
@app.route("/cut", methods=["POST"])
def cutAnAudioTest():
    time_cut_init = request.headers["time_initial"]
    time_cut_final = request.headers["time_final"]
    print(">>>> ", time_cut_init, time_cut_final)
    return {"error": "cortando"}

@app.route("/cut-and-predict", methods=["POST", "GET"]) #AQUI NO SE SI VA GET TAMBIENN ARIBA LO TIENES
def cutAnAudio():
    audio_file = request.files["file"]
    time_cut_init = request.headers["time_initial"]
    time_cut_final = request.headers["time_final"]
    print(audio_file, ">>>> ", time_cut_init, time_cut_final)
    
    file_name = ""+str(random.randint(0, 100000))+".wav"
    print(file_name)
    audio_file.save(file_name)

    #We trim the audio
    cut_cmd =  "ffmpeg -i "+file_name+" -ss "+str(time_cut_init)+" -to "+str(time_cut_final)+" -c copy cut_"+file_name+""
    os.system(cut_cmd)
    file_cut = str("cut_"+file_name+"")

    # instantiate chord spotting service singleton and get prediction
    chss = Chord_Spotting_Service()
    try:
        words_objects, chords_objects = chss.predict_song(file_cut)
    except Exception as e:
        print("Eliminando archivos creados por exception..."+ e)
        shutil.rmtree("separate")
        os.remove(file_cut)
        os.remove(file_name)
        return {"Error": "Error en la predicción"}
    else:
        print("Codificando json...")
        words_json = json.dumps(words_objects, cls=MyEncoder)
        chords_json = json.dumps(chords_objects, cls=MyEncoder)
        shutil.rmtree("separate")
        os.remove(file_cut)
        os.remove(file_name)
        result = {"words": words_json, "chords": chords_json}
        return jsonify(result)
    finally:
        print("Prediction is done...")

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=4000, debug=True) 
