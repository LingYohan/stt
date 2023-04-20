import gradio as gr
import nemo
import nemo.collections.asr as nemo_asr
from pydub import AudioSegment
from pydub.utils import make_chunks
import numpy
import librosa
import tempfile
import soundfile
import uuid
import os
 
model = nemo_asr.models.EncDecCTCModelBPE.restore_from("sf_en_v1.nemo")
SAMPLE_RATE= 16000

def reformat_audio(file):
    data, sr = librosa.load(file)
    if sr!=16000:
        data = librosa.resample(data, orig_sr=sr, target_sr=16000)
    data = librosa.to_mono(data)
    return data
    

    
def transcribe(audio):
    '''
    Speech to text fxn
    '''
    audio_data = reformat_audio(audio)

    
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = os.path.join(tmpdir, f'audio_{uuid.uuid4()}.wav')
        soundfile.write(audio_path, audio_data, SAMPLE_RATE)
        transcriptions = model.transcribe([audio_path])
        transcriptions = transcriptions[0]


    print(transcriptions)

    return transcriptions

gradio_ui = gr.Interface(
    fn=transcribe, 
    title = "Speech Recognition",
    inputs=gr.Audio(source="microphone", type="filepath"),
    outputs="text")

gradio_ui.launch()




