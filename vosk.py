# tested with VOSK 0.3.15
import vosk
import librosa
import numpy
import pandas
import sys
import os
import math
import json
import codecs
import ffmpeg
import delegator
import moviepy.editor as mp



def convert_into_audio(file):
    clip = mp.VideoFileClip(r"{}".format(file))
    name = file.split('.')[0] 
    try:
        clip.audio.write_audiofile(r"{}.mp3".format(name))
        return str(f"{name}.mp3")
    except:
        print("Error occured at file conversion!")

def extract_words(res):
   jres = json.loads(res)
   if not 'result' in jres:
       return []
   words = jres['result']
   return words

def transcribe_words(recognizer, bytes):
  
    results = []

    chunk_size = 4000
    for chunk_no in range(math.ceil(len(bytes)/chunk_size)):
        start = chunk_no*chunk_size
        end = min(len(bytes), (chunk_no+1)*chunk_size)
        data = bytes[start:end]

        if recognizer.AcceptWaveform(data):
            words = extract_words(recognizer.Result())
            results += words
    results += extract_words(recognizer.FinalResult())

    return results

def main():

    video_path = sys.argv[1]

    #Converting video into wav format to extract curse words
    audio_path = convert_into_audio(video_path)
    generated_audio = audio_path.split('.')[0]
    print(audio_path)

    #listing curse words
    words_list=[]
    muteTimeList = []
    
    with codecs.open("words_final.txt", "r") as f0:
        sentences_lines=f0.read().split()
        for sentences in sentences_lines:
            words_list.append(sentences)

    # print(words_list)
    vosk.SetLogLevel(-1)

    model_path = 'vosk-model-small-en-us-0.15'
    sample_rate = 16000

    audio, sr = librosa.load(audio_path, sr=16000)

    # convert to 16bit signed PCM, as expected by VOSK
    int16 = numpy.int16(audio * 32768).tobytes()

    if not os.path.exists(model_path):
        raise ValueError(f"Could not find VOSK model at {model_path}")

    model = vosk.Model(model_path)
    recognizer = vosk.KaldiRecognizer(model, sample_rate)

    res = transcribe_words(recognizer, int16)
    df = pandas.DataFrame.from_records(res)
    df = df.sort_values('start')
    curse = df.loc[df['word'].isin(words_list)]
    
    for index, row in curse.iterrows():
      muteTimeList.append("volume=enable='between(t," + format(row['start'], '.3f') + "," + format(row['end'], '.3f') + ")':volume=0")
      
    if len(muteTimeList) > 0:
      ffmpegCmd = "ffmpeg -y -i \"" + f"{audio_path}" + "\"" + \
                      " -af \"" + ",".join(muteTimeList) + "\"" \
                      " \"" + f"{generated_audio}_clean.mp3" + "\""
      ffmpegResult = delegator.run(ffmpegCmd, block=True)

    #For saving curse words time duration in video file
    # df.to_csv(out_path, index=False)
    # print('Curse words segments saved to', out_path)
    # print (curse)



if __name__ == '__main__':
    main()