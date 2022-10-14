import librosa
import madmom
from madmom.features.beats import RNNBeatProcessor, MultiModelSelectionProcessor
from scipy import signal
import numpy as np
import numpy as np2

def peak_picking(beat_times, total_samples, kernel_size, offset):

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

def analyze(data_audio, sample_rate):

    data_result = {}

    # sample rate
    data_result['sample_rate'] = sample_rate

    # getting duration in seconds
    data_result['duration'] = librosa.get_duration(y=data_audio, sr=sample_rate)


    rnn_processor = RNNBeatProcessor(post_processor=None)
    predictions = rnn_processor(data_audio)
    mm_processor = MultiModelSelectionProcessor(num_ref_predictions=None)
    beats = mm_processor(predictions)

    data_result['beat_samples'] = peak_picking(beats, len(data_audio), 5, 0.01)

    if len(data_result['beat_samples']) < 3:
        data_result['beat_samples'] = peak_picking(beats, len(data_audio), 25, 0.01)

    if data_result['beat_samples'] == []:
        data_result['beat_samples'] = [0]

    data_result['number_of_beats'] = len(data_result['beat_samples'])

    return data_result

def preprocessing_audio_in(rate, data, fft_size = 16384):
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