from pydub import AudioSegment
from pydub.silence import split_on_silence
import glob, os

mypath = "C:/audio/"
# Search all mp3 in folder
myfiles = glob.glob(os.path.join(mypath, "*.mp3"))

for f in myfiles:
    filename = f[f.rfind("\\")+1:f.rfind(".mp3")]
    print("Loading... %s" %f)
    sound_file = AudioSegment.from_mp3(f)
    print("Splitting...")
    audio_chunks = split_on_silence(sound_file, 
        # min seconds of silence
        min_silence_len=500,
        # min. silence in dBFS (smaller is more silent)
        silence_thresh=-48
    )

    import os
    try:
        os.makedirs(mypath + "//split")
    except OSError:
        pass

    for i, chunk in enumerate(audio_chunks):
        out_file = mypath + "//split//"+filename+"_chunk{0}.mp3".format(i)
        print("exporting %s" %out_file)
        chunk.export(out_file, format="mp3")
