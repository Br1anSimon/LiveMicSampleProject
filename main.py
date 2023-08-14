import numba
import sounddevice as sd
import wavio as wv
import whisper
import multiprocessing
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings
from numba import jit

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


@numba.jit(nopython=True, parallel=True)
def record(conn):
    # Sampling frequency
    freq = 44100
    # Recording duration
    duration = 5
    print('Recording')

    while True:
        newmsg = False
        conn.send(newmsg)
        # Start recorder with the given values
        # of duration and sample frequency
        recording = sd.rec(int(duration * freq),
                           samplerate=freq, channels=2)

        # Record audio for the given number of seconds
        sd.wait()
        # i wonder if i can bounce between two files so i don't lose words

        # Convert the NumPy array to audio file
        wv.write("recording0.wav", recording, freq, sampwidth=2)
        newmsg = True
        conn.send(newmsg)


def transcribe(conn, model):
    while True:
        while True:
            newmsg = conn.recv()
            if newmsg:
                break

        audio = whisper.load_audio("recording0.wav")
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions(language='en', fp16=False)
        result = whisper.decode(model, mel, options)

        print(result.text)


if __name__ == "__main__":
    model = whisper.load_model("small.en")

    to_mic, to_whisper = multiprocessing.Pipe()
    mic = multiprocessing.Process(target=record, args=(to_whisper,))
    write = multiprocessing.Process(target=transcribe, args=(to_mic, model))
    mic.start()
    write.start()
    mic.join()
    write.join()
