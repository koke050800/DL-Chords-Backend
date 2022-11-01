'Parte de este código es tomado de Valerio Velardo'

import json
import librosa
import madmom
from madmom.features.beats import RNNBeatProcessor, MultiModelSelectionProcessor
import tensorflow as tf
from scipy import signal
import numpy as np
import numpy as np2
import requests
import time
from functions.api_secrets import API_KEY_ASSEMBLYAI
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

SAVED_MODEL_PATH = "src/functions/DLCords1_CPU.h5" 
upload_endpoint = 'https://api.assemblyai.com/v2/upload'
transcript_endpoint = 'https://api.assemblyai.com/v2/transcript'

headers_auth_only = {'authorization': API_KEY_ASSEMBLYAI}

headers = {
    "authorization": API_KEY_ASSEMBLYAI,
    "content-type": "application/json"
}

CHUNK_SIZE = 5_242_880*5  

class _Chord_Spotting_Service:
    """Singleton class for keyword spotting inference with trained models.
    :param model: Trained model
    """

    model = None
    _instance = None

    def peak_picking(self,beat_times, total_samples, kernel_size, offset):

        # smoothing the beat function
        cut_off_norm = len(beat_times)/total_samples*100/2
        b, a = signal.butter(1, cut_off_norm)
        beat_times = signal.filtfilt(b, a, beat_times)

        # creating a list of samples for the rnn beats
        beat_samples = np2.linspace(0, total_samples, len(beat_times), endpoint=True, dtype=int)

        n_t_medians = signal.medfilt(beat_times, kernel_size=kernel_size)
        offset = 0.01
        peaks = []

        for i in range(len(beat_times)-1):
            if beat_times[i] > 0:
                if beat_times[i] > beat_times[i-1]:
                    if beat_times[i] > beat_times[i+1]:
                        if beat_times[i] > (n_t_medians[i] + offset):
                            peaks.append(int(beat_samples[i]))
        return peaks

    def analyze(self,data_audio, sample_rate):

        data_result = {}

        # sample rate
        data_result['sample_rate'] = sample_rate

        # getting duration in seconds
        data_result['duration'] = librosa.get_duration(y=data_audio, sr=sample_rate)


        rnn_processor = RNNBeatProcessor(post_processor=None)
        predictions = rnn_processor(data_audio)
        mm_processor = MultiModelSelectionProcessor(num_ref_predictions=None)
        beats = mm_processor(predictions)

        data_result['beat_samples'] = self.peak_picking(beats, len(data_audio), 5, 0.01)

        if len(data_result['beat_samples']) < 3:
            data_result['beat_samples'] = self.peak_picking(beats, len(data_audio), 25, 0.01)

        if data_result['beat_samples'] == []:
            data_result['beat_samples'] = [0]

        data_result['number_of_beats'] = len(data_result['beat_samples'])

        return data_result

    def preprocessing_audio_in(self,rate, data, fft_size = 16384):
        """
        Convert the input audio sampled at the input rate
        to a list of HPCP vectors computed using the input fft_size
        (effectively outputing int(len(data)/fft_size)) HPCP vectors
        """
        output_samples = []
        for i in range(int(len(data)/fft_size)):
            ###Computing the DFT by taking a fragment of the audio 
            dft = np.fft.fft(data[fft_size*i:fft_size*(i+1)])
            ### Computiong the Harmonic pitch class profile
            HPCP = []
            f_ref = 130.80
            M = [round(12*np.log2(rate*l/(fft_size*f_ref))) %12 if l > 0 else -1 for l in range(int(fft_size/2))]
            M = np.array(M)
            for p in range(12):
                val = np.sum((np.absolute(dft[:int(fft_size/2)])**2)* (M == p).astype(int) )
                HPCP.append(val)
            HPCP = [x/sum(HPCP) for x in HPCP]
            output_sample = HPCP
            output_samples.append(output_sample)
        
        return output_samples

    def convert_to_wav_and_filter_audio(self,path):
        split_cmd = "spleeter separate -p spleeter:2stems -o separate/ "+path+""
        #split_cmd = "spleeter --help"
        print(split_cmd)
        import os;os.system(split_cmd)

    def predict_chords(self,path_of_chords):
        print("Prediccion de acordes en curso...")
        data_audio, sample_rate = librosa.load(path_of_chords, sr=44100)
        samples_for_beat = self.analyze(data_audio, sample_rate)['beat_samples']
        seconds = []
        secondFinalOfChord = int()
        data_divide_in_chords = []
        
        for i in range(len(samples_for_beat)):
            if(i== len(samples_for_beat)-1):
                data_divide_in_chords.append(data_audio[samples_for_beat[i]:len(data_audio)])
            else:
                data_divide_in_chords.append(data_audio[samples_for_beat[i]:samples_for_beat[i+1]])
                
        pitchs_of_all_data = []
            
        for i in data_divide_in_chords:
            secondFinalOfChord = secondFinalOfChord  + int(len(i))
            seconds.append(secondFinalOfChord/44100)        
            pitch_s = self.preprocessing_audio_in(rate=sample_rate, data=i)
            if(len(pitch_s)>0):
                pitchs_of_all_data.append(pitch_s[0])#Solo enviamos el primer pitch
            
        #a1_data, a1_rate = librosa.load(path_of_chords='C:\\Users\\UPIIZ 35\\Documents\\DLChordsNotebook\\data\\Guitar_Only\\a\\a1.wav', sr=44100)
        #pitch_clean = preprocessing_audio_in(a1_rate, a1_data)
        #print(pitchs_of_all_data)

        prediction = self.model.predict(pitchs_of_all_data)
        prediction = np.argmax(prediction, axis=1).astype(int)
        
        labels_dict_reverse = {
        0:'a',
        1:'am',
        2:'bm',
        3:'c',
        4:'d',
        5:'dm',
        6:'e',
        7:'em',
        8:'f',
        9:'g'
        }

        prediction_in_string = []
        for i in prediction:   
            prediction_in_string.append(labels_dict_reverse[i])   
        
        #print(">>> ", seconds)
        #print(prediction, prediction_in_string)


        class Chord:
            def __init__(self, chord_result, time_init, time_final):
                self.chord_result = chord_result
                self.time_init = time_init
                self.time_final = time_final
            def __str__(self):
                return json.dumps(dict(self), ensure_ascii=False)
            def __repr__(self):
                return str(self.__dict__)

        
        chords_objects = []

        for i in range(len(prediction_in_string)):
            if(i==0):
                chords_objects.append(Chord(str(prediction_in_string[i]), 0.00, round(seconds[0], 2)))
            else:
                chords_objects.append(Chord(str(prediction_in_string[i]), round(seconds[i-1], 2), round(seconds[i],2)))
            
        return chords_objects

    def predict_song(self,path):
        print("Convirtiendo y filtrando audio...")
        print(path)
        self.convert_to_wav_and_filter_audio(path)#convertimos audio y filtramos

        path_whitout_extension = path.rsplit('.', 1)[0]
        #print(path_whitout_extension)
        path_of_chords = "separate/" + path_whitout_extension + "/accompaniment.wav"#ruta de archivo con extension .wav, solo con la melodia, sin voz.
        path_of_vocal = "separate/" + path_whitout_extension + "/vocals.wav" #cambiar el idioma, 
        #print(path_of_chords)

        
        return self.data_words_to_object(self.get_transcription_result_url(self.upload(path_of_vocal),False)), self.predict_chords(path_of_chords) #Predecir acordes y la letra

    # PARA RECONOCER LA LETRA USAMOS LAS SIGUIENTES FUNCIONES 

    def upload(self,filename):
        def read_file(filename):
            with open(filename, 'rb') as f:
                while True:
                    data = f.read(CHUNK_SIZE)
                    if not data:
                        break
                    yield data

        upload_response = requests.post(upload_endpoint, headers=headers_auth_only, data=read_file(filename))
        return upload_response.json()['upload_url']

    def transcribe(self,audio_url, language):
        if language == True:#Idioma Inglés
            transcript_request = {
                'audio_url': audio_url,

            } 
        else: #Iidioma Español
            transcript_request = {
                'audio_url': audio_url,
                "language_code": "es"

            } 

        transcript_response = requests.post(transcript_endpoint, json=transcript_request, headers=headers)
        return transcript_response.json()['id']
    
    def poll(self, transcript_id):
        polling_endpoint = transcript_endpoint + '/' + transcript_id
        polling_response = requests.get(polling_endpoint, headers=headers)
        return polling_response.json()
    
    def get_transcription_result_url(self, url,language):
        print("Predicción de letra en curso...")
        transcribe_id = self.transcribe(url,language)
        while True:
            data = self.poll(transcribe_id)
            if data['status'] == 'completed':
                return data
            elif data['status'] == 'error':
                print("EL ERROR DE LA LETRA"+data['status'])
                return data
                
            print("waiting for 8 seconds")
            time.sleep(8)
    
    def data_words_to_object(self,data):
        class Words:
            def __init__(self, word_result, time_init, time_final):
                self.word_result = word_result
                self.time_init = time_init
                self.time_final = time_final
            def __repr__(self):
                return str(self.__dict__)
            #print(data['words'][0]['text'])
            #print(len(data['words']))
        words_objects = []

        for i in range(len(data['words'])):
            words_objects.append(Words(data['words'][i]['text'],round(float(data['words'][i]['start'])/1000,2),round(float(data['words'][i]['end'])/1000, 2)))
        
        return words_objects   

def Chord_Spotting_Service():
        """Factory function for Keyword_Spotting_Service class.
        :return _Keyword_Spotting_Service._instance (_Keyword_Spotting_Service):
        """

        # ensure an instance is created only the first time the factory function is called
        if _Chord_Spotting_Service._instance is None:
            _Chord_Spotting_Service._instance = _Chord_Spotting_Service()
            _Chord_Spotting_Service.model = tf.keras.models.load_model(SAVED_MODEL_PATH)
        return _Chord_Spotting_Service._instance

    
if __name__ == "__main__":

    # create 2 instances of the keyword spotting service
    kss = Chord_Spotting_Service()
    kss1 = Chord_Spotting_Service()

    # check that different instances of the keyword spotting service point back to the same object (singleton)
    assert kss is kss1

    # make a prediction
    words_objects, chords_objects = kss.predict_song("audio_example.mp3")
    #print(words['words'])
    for word_object in words_objects:  
        print("WORD>>", word_object.word_result,"     TIEMPOS>> ", word_object.time_init, "   -   ", word_object.time_final)

    for chord_object in chords_objects:
        if(len(chord_object.chord_result)==2):
            print("CHORDISTO>>", chord_object.chord_result,"    TIEMPOS>> ", chord_object.time_init, "   -   ", chord_object.time_final)
        else:
            print("CHORDISTO>>", chord_object.chord_result,"     TIEMPOS>> ", chord_object.time_init, "   -   ", chord_object.time_final)
    
