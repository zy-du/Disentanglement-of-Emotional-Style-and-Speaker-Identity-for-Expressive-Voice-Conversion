import hydra
import hydra.utils as utils

from utils import read_hdf5

from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

import soundfile as sf

from model_encoder import Encoder, Encoder_lf0
from model_encoder import EmoEncoder
from model_decoder import Decoder_ac
from model_encoder import SpeakerEncoder as Encoder_spk
import os
from os.path import join
import random

from glob import glob
import subprocess
from spectrogram import logmelspectrogram
import kaldiio

import resampy
import pyworld as pw

import shutil


def save_one_file(save_path, arr):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, arr)


def save_logmel(save_root, wav_name, mel):
    mel_save_path = f'{save_root}/{wav_name}.npy'
    save_one_file(mel_save_path, mel)

    return


def select_wavs(paths, min_dur=2, max_dur=8):
    pp = []
    for p in paths:
        x, fs = sf.read(p)
        if len(x) / fs >= min_dur and len(x) / fs <= 8:
            pp.append(p)
    return pp

def load_mean_std(stat_path,spk ):
    stat_dir = stat_path + '/' + spk +'/' + spk + '_stats.npz'
    spk_stats = np.load(stat_dir)
    logf0s_mean = spk_stats['log_f0s_mean']
    logf0s_std = spk_stats['log_f0s_std']
    #mel_mean = spk_stats['mels_mean']
    #mel_std = spk_stats['mels_std']
    return logf0s_mean, logf0s_std



def extract_logmel(wav_path, mean, std, sr=16000):
    # wav, fs = librosa.load(wav_path, sr=sr)
    wav, fs = sf.read(wav_path)
    if fs != sr:
        wav = resampy.resample(wav, fs, sr, axis=0)
        fs = sr
    # wav, _ = librosa.effects.trim(wav, top_db=15)
    # duration = len(wav)/fs
    assert fs == 16000
    peak = np.abs(wav).max()
    if peak > 1.0:
        wav /= peak
    mel = logmelspectrogram(
        x=wav,
        fs=fs,
        n_mels=80,
        n_fft=400,
        n_shift=160,
        win_length=400,
        window='hann',
        fmin=80,
        fmax=7600,
    )
    mel = (mel - mean) / (std + 1e-8)
    tlen = mel.shape[0]
    frame_period = 160 / fs * 1000
    # frame_period = fs
    f0, timeaxis = pw.dio(wav.astype('float64'), fs, frame_period=frame_period)

    f0 = pw.stonemask(wav.astype('float64'), f0, timeaxis, fs)
    f0 = f0[:tlen].reshape(-1).astype('float32')
    #nonzeros_indices = np.nonzero(f0)
    #lf0 = f0.copy()
    #lf0[nonzeros_indices] = np.log(f0[nonzeros_indices])  # for f0(Hz), lf0 > 0 when f0 != 0

    #mean, std = np.mean(lf0[nonzeros_indices]), np.std(lf0[nonzeros_indices])
    #lf0[nonzeros_indices] = (lf0[nonzeros_indices] - mean) / (std + 1e-8)
    return mel, f0


def pitch_conversion(f0, mean_log_target, std_log_target):
    nonzeros_indices = np.nonzero(f0)
    lf0 = f0.copy()
    lf0[nonzeros_indices] = np.log(f0[nonzeros_indices])
    lf0[nonzeros_indices] = lf0[nonzeros_indices] * std_log_target + mean_log_target
    #lf0[nonzeros_indices] =((lf0[nonzeros_indices] - mean_log_src) / std_log_src) * std_log_target + mean_log_target
    mean, std = np.mean(lf0[nonzeros_indices]), np.std(lf0[nonzeros_indices])
    lf0[nonzeros_indices] = (lf0[nonzeros_indices] - mean) / (std + 1e-8)
    return lf0

@hydra.main(config_path="./config/convert.yaml")
def convert(cfg):

    src_spks_list = cfg.src_spks
    src_spks = src_spks_list.split(",")
    trg_spks_list = cfg.trg_spks
    trg_spks = trg_spks_list.split(",")
    mode_list = cfg.mode
    mode1 = mode_list.split(",")
    stat_path = cfg.stat_path
    data_path = cfg.data_path
    trg_path = cfg.trg_path
    spk_num = len(src_spks)
    emotion = cfg.emotion
    out_path = cfg.out_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(**cfg.model.encoder)
    encoder_lf0 = Encoder_lf0()
    encoder_spk = Encoder_spk()

    encoder_emo = EmoEncoder()

    decoder = Decoder_ac(dim_neck=64)
    encoder.to(device)
    encoder_lf0.to(device)
    encoder_spk.to(device)
    encoder_emo.to(device)
    decoder.to(device)

   
    checkpoint_path = cfg.checkpoint



    print("Load checkpoint from: {}:", checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)

    encoder.load_state_dict(checkpoint["encoder"])
    encoder_spk.load_state_dict(checkpoint["encoder_spk"])
    encoder_emo.load_state_dict(checkpoint["encoder_emo"])
    encoder_lf0.load_state_dict(checkpoint["encoder_lf0"])
    decoder.load_state_dict(checkpoint["decoder"])

    encoder.eval()
    encoder_emo.eval()
    encoder_spk.eval()
    decoder.eval()

    mel_stats = np.load(cfg.mel_state_path)
    
    mean = mel_stats[0]
    std = mel_stats[1]

    for i in range(spk_num):
        src_spk = src_spks[i]
        trg_spk = trg_spks[i]
        src_wav_paths = sorted(glob(join(data_path, f'{src_spk}/{emotion}/test/*.wav')))
        src_wav_paths = [item.replace('\\', '/') for item in src_wav_paths]

        tar_wav_paths = sorted(glob(join(trg_path, f'{trg_spk}/{emotion}/test/*.wav')))
        tar_wav_paths = [item.replace('\\', '/') for item in tar_wav_paths]

        print('len(src):', len(src_wav_paths), 'len(tar1):', len(tar_wav_paths), )

        # tmp = cfg.checkpoint.split('/')

        out_d = out_path + '/' + mode1[i // 2] + '/' + src_spk + 'to' + trg_spk
        out_dir = Path(utils.to_absolute_path(out_d))
        out_dir.mkdir(exist_ok=True, parents=True)

        feat_writer = kaldiio.WriteHelper("ark,scp:{o}.ark,{o}.scp".format(o=str(out_dir) + '/feats.1'))

        trg_logf0s_mean, trg_logf0s_std = load_mean_std(stat_path, trg_spk)



        for i, src_wav_path in tqdm(enumerate(src_wav_paths, 1)):
            #mel, lf0 = extract_logmel(src_wav_path, mean, std)
            mel, scr_f0 = extract_logmel(src_wav_path, mean, std)
            lf0 = pitch_conversion(scr_f0, trg_logf0s_mean, trg_logf0s_std)
            ref_wav_path = tar_wav_paths[-1]
            ref_mel, ref_lf0 = extract_logmel(ref_wav_path, mean, std)


            print(src_wav_path, ref_wav_path)

            #shutil.copy(src_wav_path, out_d)


            mel = torch.FloatTensor(mel.T).unsqueeze(0).to(device)
            #ref_lf0_new  = torch.FloatTensor( ref_lf0_new ).unsqueeze(0).to(device)
            lf0 = torch.FloatTensor(lf0).unsqueeze(0).to(device)

            ref_mel = torch.FloatTensor(ref_mel.T).unsqueeze(0).to(device)

            out_filename = os.path.basename(src_wav_path).split('.')[0]
            tar_wav_path = ref_wav_path.split('_')[0]+ '_' + os.path.basename(src_wav_path).split('_')[-1]


            with torch.no_grad():
                z, _, _, _ = encoder.encode(mel)
                #lf0_embs = encoder_lf0( ref_lf0_new )
                lf0_embs = encoder_lf0(lf0)

                spk_embs = encoder_spk(ref_mel)
                emo_embs = encoder_emo(ref_mel)
                output = decoder(z, lf0_embs, spk_embs, emo_embs)
                logmel = output.squeeze(0).cpu().numpy()

                # feat_writer[out_filename] = logmel
                feat_writer[out_filename] = logmel
                #feat_writer[out_filename + '_src'] = mel.squeeze(0).cpu().numpy().T
                # feat_writer[out_filename + '_ref'] = ref_mel.squeeze(0).cpu().numpy().T
                #feat_writer[out_filename + '_ref'] = ref_mel.squeeze(0).cpu().numpy().T

        feat_writer.close()
        # subprocess.call(['cp', src_wav_path, out_dir])

        print('synthesize waveform...')

        vocoder_path = cfg.vocoder_path

        cmd = ['parallel-wavegan-decode', '--checkpoint', \
               vocoder_path, \
               '--feats-scp', f'{str(out_dir)}/feats.1.scp', '--outdir', str(out_dir)]
        subprocess.call(cmd)


if __name__ == "__main__":
    
    convert()
