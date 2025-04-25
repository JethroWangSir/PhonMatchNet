import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import wavfile
from speech_embedding import GoogleSpeechEmbedder
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

fs = 16000
debug = False

def create_npy(data, wav_dir, save_dir, desc="libriphrase"):
        data = data.sort_values(by='duration').reset_index(drop=True)
        data['wav'] = data['wav'].apply(lambda x: os.path.join(wav_dir, x))
        wav_list = data['wav'].values
        maxlen_a = int((int(data['duration'].values[-1] / 0.5) + 1 ) * fs / 2)
        EMB = GoogleSpeechEmbedder()

        for i in tqdm(range(len(wav_list)), desc=desc):
                file_dirs = os.path.splitext(wav_list[i])[0]
                file_dirs = file_dirs.split("/")
                filepath = file_dirs[-1] + '.npy'
                dirpath = os.path.join(*file_dirs[3:-1])
                dirpath = os.path.join(save_dir, dirpath)
                filepath = os.path.join(dirpath, filepath)
                
                if debug:
                        print()
                        print("original path : {}".format(wav_list[i]))
                        print("new path      : {}".format(filepath))
                        print("[WARNING] debug flag has been set. " +
                              "Please confirm the new path before proceeding. " +
                              "If you intend to save embeddings in a new location, ensure that the debug flag is set to False.")
                        return
                        
                if (os.path.exists(filepath)):
                        continue
                if not(os.path.exists(dirpath)):
                        os.makedirs(dirpath)
                
                x = [np.array(wavfile.read(wav_list[i])[1]).astype(np.float32) / 32768.0]
                # Padding: followed the official code
                x = pad_sequences(np.array(x), maxlen=maxlen_a, value=0.0, padding='post', dtype=x[0].dtype)
                emb = EMB(x).numpy()
                np.save(filepath, emb)


def preprocess_libriphrase(wav_dir = '/share/nas169/albertali/Libriphrase/LibriPhrase_data',
                           csv_dir = '/share/nas169/albertali/Libriphrase/LibriPhrase_data',
                           save_dir = '/share/nas169/jethrowang/DB/',
                           train_csv = ['train100h', 'train_360h'],
                           test_csv = ['train_500h',],
                           train = True,
                           ):

        data = pd.DataFrame(columns=['wav_label', 'wav', 'text', 'duration', 'label', 'type'])

        for db in train_csv if train else test_csv:
                csv_list = [str(x) for x in Path(csv_dir).rglob('*' + db + '*word*')]
                for n_word in csv_list:
                        print(">> processing : {} ".format(n_word))
                        df = pd.read_csv(n_word)
                        # Split train dataset to match & unmatch case
                        anc_pos = df[['anchor_text', 'anchor', 'anchor_text', 'anchor_dur']].copy()
                        anc_neg = df[['anchor_text', 'anchor', 'comparison_text', 'anchor_dur', 'target', 'type']].copy()
                        com_pos = df[['comparison_text', 'comparison', 'comparison_text', 'comparison_dur']].copy()
                        com_neg = df[['comparison_text', 'comparison', 'anchor_text', 'comparison_dur', 'target', 'type']].copy()
                        anc_pos.columns = ['wav_label', 'anchor', 'anchor_text', 'anchor_dur']
                        com_pos.columns = ['wav_label', 'comparison', 'comparison_text', 'comparison_dur']
                        anc_pos.loc[:, 'label'] = 1
                        anc_pos.loc[:, 'type'] = df['type']
                        com_pos.loc[:, 'label'] = 1
                        com_pos.loc[:, 'type'] = df['type']
                        # Concat
                        data = pd.concat([data, anc_pos.rename(columns={y: x for x, y in zip(data.columns, anc_pos.columns)})], ignore_index=True)
                        data = pd.concat([data, anc_neg.rename(columns={y: x for x, y in zip(data.columns, anc_neg.columns)})], ignore_index=True)
                        data = pd.concat([data, com_pos.rename(columns={y: x for x, y in zip(data.columns, com_pos.columns)})], ignore_index=True)
                        data = pd.concat([data, com_neg.rename(columns={y: x for x, y in zip(data.columns, com_neg.columns)})], ignore_index=True)


        create_npy(data, wav_dir, save_dir, desc="libriphrase")
        return


def preprocess_google(wav_dir = '/share/nas169/albertali/yiting/kws_google/data2',
                      save_dir = '/share/nas169/jethrowang/DB/',
                      target_list = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'],
                      ):

        data = pd.DataFrame(columns=['wav', 'text', 'duration', 'label'])

        print(">> Make dataset from {}".format(wav_dir))
        target_dict = {}
        idx = 0
        for target in target_list:    
                print(">> Extract from {}".format(target))
                wav_list = [str(x) for x in Path(os.path.join(wav_dir, target)).rglob('*.wav')]
                for wav in wav_list:
                        anchor_text = wav.split('/')[-2].lower()
                        duration = float(wavfile.read(wav)[1].shape[-1])/fs
                        for comparison_text in target_list:
                                comparison_text = comparison_text.replace('_', ' ')
                                label = 1 if anchor_text == comparison_text else 0
                                target_dict[idx] = {
                                        'wav': wav,
                                        'text': comparison_text,
                                        'duration': duration,
                                        'label': label
                                }
                                idx += 1

        data = pd.concat([data, pd.DataFrame.from_dict(target_dict, 'index')], ignore_index=True)
        create_npy(data, wav_dir, save_dir, desc="google")
        return


def preprocess_qualcomm(wav_dir = '/share/nas169/albertali/Zero-shot-phonmatchnet/Zero-shot_CL-KWS_202405_v3/qualcomm_keyword_speech_dataset',
                        save_dir = '/share/nas169/jethrowang/DB/',
                        target_list=['hey_android', 'hey_snapdragon', 'hi_galaxy', 'hi_lumina'],
                        ):

        data = pd.DataFrame(columns=['wav', 'text', 'duration', 'label'])

        print(">> Make dataset from {}".format(wav_dir))
        target_dict = {}
        idx = 0
        for target in target_list:    
                print(">> Extract from {}".format(target))
                wav_list = [str(x) for x in Path(os.path.join(wav_dir, target)).rglob('*.wav')]
                for wav in wav_list:
                        anchor_text = wav.split('/')[-2].lower()
                        duration = float(wavfile.read(wav)[1].shape[-1])/fs
                        for comparison_text in target_list:
                                comparison_text = comparison_text.replace('_', ' ')
                                label = 1 if anchor_text == comparison_text else 0
                                target_dict[idx] = {
                                        'wav': wav,
                                        'text': comparison_text,
                                        'duration': duration,
                                        'label': label
                                }
                                idx += 1
                                
        data = pd.concat([data, pd.DataFrame.from_dict(target_dict, 'index')], ignore_index=True)
        create_npy(data, wav_dir, save_dir, desc="qualcomm")
        return


def main():
        save_dir='/share/nas169/jethrowang/DB/npy'
        preprocess_libriphrase(save_dir=save_dir, train=True)
        preprocess_libriphrase(save_dir=save_dir, train=False)
        preprocess_google(save_dir=save_dir)
        preprocess_qualcomm(save_dir=save_dir)


main()