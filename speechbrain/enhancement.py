from interfaces import SepformerSeparation as separator
import torchaudio
import time
from pydub import AudioSegment
import glob
import io
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


model = separator.from_hparams(source="speechbrain/sepformer-whamr-enhancement", savedir='pretrained_models/sepformer-whamr-enhancement', run_opts={"device":"cuda"})


est_sources = model.separate_file(path='/home/auishik/speech_enhancement/musical_audio.wav') 

# enhanced_audio_bytes = io.BytesIO()
# enhanced_audio_bytes
torchaudio.save('enhanced.wav', est_sources[:, :, 0].detach().cpu(), sample_rate=8000, format="wav")

print('done')

