import re
import numpy as np
import json
import pandas as pd
from tqdm import tqdm
import os
import sys
import logging
from collections import Counter
from tokenizers import Tokenizer, SentencePieceBPETokenizer
import torch
import pickle
from src.utils import Vocab

def normalize_name(name):
    name = name.lower()
    name = re.sub(r"[.,\[\]\/#!$%\^\*;:{}=\_`~()@<>]", " ", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name

def filter_title(title, songs, min_token_len=3, min_title_len=3, min_tracklist_len=10):
    n_tracks = len(songs)
    tokens = title.split(' ')
    mean_token_len = np.array([len(i) for i in tokens]).mean() if len(tokens) else 0
    
    if mean_token_len >= min_token_len and len(tokens) >= min_title_len and n_tracks >= min_tracklist_len:
        return True
    else:
        return False

def get_artists_set(tracks, melon_songs):
    artist_ids = []
    for track in tracks:
        artist_id = melon_songs[melon_songs['id']==track]['artist_id_basket'].values
        if not len(artist_id) == 1:
            raise ValueError("Wrong artist id retrieval: got {}(not 1) tracks w/ track ID!".format(len(artist_id))) 
        artist_ids += artist_id[0]
    return list(set(artist_ids))

def load_and_filter(dataset_name, dataset_dir, filtered_dir, min_token_len=3, min_title_len=3, min_tracklist_len=10):
    tqdm.pandas()

    if not (dataset_name == 'melon' or dataset_name == 'mpd'):
        raise ValueError("Please insert correct dataset name: 'melon' or 'mpd'.")

    if dataset_name == 'melon':
        melon_train = pd.read_json(os.path.join(dataset_dir, "train.json"))
        melon_val = pd.read_json(os.path.join(dataset_dir, "val.json"))
        melon_test = pd.read_json(os.path.join(dataset_dir, "test.json"))
        melon_songs = pd.read_json(os.path.join(dataset_dir, "song_meta.json"))

        melon_playlist = pd.concat([melon_train, melon_val, melon_test], axis=0)
        melon_playlist = melon_playlist[melon_playlist['plylst_title'].map(lambda r: len(r.split(' '))>0 if not r=='' else False)]
        melon_playlist = melon_playlist[melon_playlist['songs'].map(lambda r: len(r)>0)]

        logger.info("--- Melon Playlist Dataset")
        playlist = melon_playlist[['id', 'plylst_title', 'songs', 'tags']]
        playlist = playlist.rename(columns={'id': 'pid'})
        logger.info("got {} playlists in total.".format(len(playlist)))
        logger.info("Get Artist IDs...")
        playlist['artists'] = playlist['songs'].progress_map(lambda tracks: get_artists_set(tracks, melon_songs))
        
    elif dataset_name == 'mpd':
        fname = os.listdir(dataset_dir)
        logger.info("--- MPD (Million Playlist Dataset)")

        dfs = []
        for file in tqdm(fname):
            if file.startswith("mpd.slice.") and file.endswith(".json"):
                file = json.load(open(os.path.join(dataset_dir,file),'r'))
                df = pd.DataFrame.from_dict(file['playlists'])
                df = df[df['name'].map(lambda r: len(r.split(' '))>0 if not r=='' else False)]
                df['songs'] = df['tracks'].map(lambda tracks: [track['track_uri'] for track in tracks if len(tracks)>0])
                df['artists'] = df['tracks'].map(lambda tracks: list(set([track['artist_uri'] for track in tracks])))
                df = df.rename(columns={'name': 'plylst_title'})
                dfs.append(df[['pid', 'plylst_title', 'songs', 'artists']])
        playlist = pd.concat(dfs)
        logger.info("got {} playlists in total.".format(len(playlist)))
    

    logger.info("Normalize Playlist Title...")
    playlist['nrm_plylst_title'] = playlist['plylst_title'].progress_map(normalize_name)
    logger.info("Filter Playlists...")
    filtered_playlist = playlist[playlist.progress_apply(lambda row: filter_title(row['nrm_plylst_title'], row['songs'],\
                                                                min_token_len, min_title_len, min_tracklist_len), axis=1)]
    logger.info("{} playlists are retrieved.".format(len(filtered_playlist)))

    filtered_playlist_dict = filtered_playlist.to_dict(orient='records')
    torch.save(filtered_playlist_dict, os.path.join(filtered_dir, dataset_name+'_filtered.pt'))
    # filtered_playlist.to_csv(os.path.join(filtered_dir, dataset_name+'_filtered.csv'), index=False)
    logger.info("Filtered {} Dataset Saved.".format(dataset_name.upper()))


def data_split(filtered_dir, split_dir, ratio=[0.8, 0.1, 0.1]):
    # df = pd.read_csv(filtered_dir)
    data_dict = torch.load(filtered_dir)
    df = pd.DataFrame.from_dict(data_dict)
    if not len(ratio)==3:
        raise ValueError('Insert ''3'' ratio values for train/val/test dataset.')
    if any(r < 0. or r > 1. for r in ratio) or round(sum(ratio), 5) != 1:
        raise ValueError('Ratio should be values btw 0 and 1, and its sum should be 1.')
    
    min_title_len = df['nrm_plylst_title'].apply(lambda r: len(r.split(' ')) if r!=None else 0).min()
    max_title_len = df['nrm_plylst_title'].apply(lambda r: len(r.split(' ')) if r!=None else 0).max()
    if min_title_len == 0:
        raise Exception('Playlist w/ no title exists!')

    dfs = {'train': [], 'val': [], 'test':[]}
    for title_len in range(min_title_len, max_title_len+1):
        uni_len_df = df[df['nrm_plylst_title'].map(lambda r: len(r.split(' '))==title_len)]

        train, validate, test = np.split(uni_len_df.sample(frac=1, random_state=33),
                                                [int(ratio[0]*len(uni_len_df)), int((ratio[0]+ratio[1])*len(uni_len_df))])
        dfs['train'].append(train)
        dfs['val'].append(validate)
        dfs['test'].append(test)
    
    for name in dfs:
        merged_dataset = pd.concat(dfs[name])

        merged_dataset_dict = merged_dataset.to_dict(orient='records')
        torch.save(merged_dataset_dict, os.path.join(split_dir, name+'.pt'))
        # merged_dataset.to_csv(os.path.join(split_dir, name+'.csv'), index=False)
        logger.info("Filtered {} Dataset Saved: total {} playlists.".format(name.upper(), len(merged_dataset)))
    
def byte_level_BPE_train(train_dir, val_dir, out_dir, out_name, limit_alphabet, vocab_size=10000):
    # Load Dataset
    train_dict = torch.load(train_dir)
    train_df = pd.DataFrame.from_dict(train_dict)
    val_dict = torch.load(val_dir)
    val_df = pd.DataFrame.from_dict(val_dict)
    df = pd.concat([train_df, val_df], axis=0)
    # df = pd.read_csv(dataset_dir)

    # Initialize an empty tokenizer
    tokenizer = SentencePieceBPETokenizer()

    # And then train
    tokenizer.train_from_iterator(
        df['nrm_plylst_title'],
        vocab_size=vocab_size,
        min_frequency=2,
        limit_alphabet=limit_alphabet,
        show_progress=True,
        special_tokens=["<pad>", "<sos>", "<eos>", "<unk>"]
    )
        ###["<s>", "<pad>", "</s>"], "<|startoftext|>" "<|endoftext|>"

    # Save BPE model
    tokenizer.save_model(directory=out_dir, prefix=out_name)
    logger.info("BPE model trained & saved.")


def build_dictionary(dataset_dir, track_out_dir, song_out_dir, out_name):
    data_dict = torch.load(dataset_dir)
    song_list = []
    token_list = []
    for instance in data_dict:
        song_list.extend(instance['songs'])
    for instance in data_dict:
        token_list.extend(instance['nrm_plylst_title'].split())
    s_counter = Counter(song_list)
    t_counter = Counter(token_list)

    s_vocab = Vocab(list_of_tokens=list(s_counter.keys()))
    t_vocab = Vocab(list_of_tokens=list(t_counter.keys()))

    with open(os.path.join(track_out_dir, out_name + "_vocab.pkl"), mode="wb") as io:
        pickle.dump(t_vocab, io)
    
    with open(os.path.join(song_out_dir, out_name + "_vocab.pkl"), mode="wb") as io:
        pickle.dump(s_vocab, io)
    

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    logger.info("----- LOAD AND FILTER DATASETS -----")
    load_and_filter('melon', "./dataset/source/melon/", "./dataset/split/",\
                                min_token_len=3, min_title_len=3, min_tracklist_len=10)
    load_and_filter('mpd', "./dataset/source/mpd/data", "./dataset/split/",\
                                min_token_len=3, min_title_len=3, min_tracklist_len=10)

    logger.info("----- SPLIT DATASETS (TR/VA/TE) -----")
    logger.info("--- Melon Playlist Dataset")
    data_split("./dataset/split/melon_filtered.pt", "./dataset/split/melon/", ratio=[0.8, 0.1, 0.1])
    logger.info("--- MPD (Million Playlist Dataset)")
    data_split("./dataset/split/mpd_filtered.pt", "./dataset/split/mpd/", ratio=[0.8, 0.1, 0.1])
    
    logger.info("----- BUILD TRACK DICTIONARY -----")
    logger.info("--- Melon Playlist Dataset")
    build_dictionary("./dataset/split/melon_filtered.pt", './dataset/tokenizer/title_split', './dataset/tokenizer/track', 'melon')
    logger.info("--- MPD (Million Playlist Dataset)")
    build_dictionary("./dataset/split/mpd_filtered.pt", './dataset/tokenizer/title_split', './dataset/tokenizer/track', 'mpd')

    # logger.info("----- TRAIN SENTENCE LEVEL BPE Work in Progress -----")
    # logger.info("--- Melon Playlist Dataset")
    # byte_level_BPE_train('./dataset/split/melon/train.pt', './dataset/split/melon/val.pt',\
    #                         './dataset/tokenizer/title_bpe', 'melon', vocab_size=20000, limit_alphabet=6000)
    # logger.info("--- MPD (Million Playlist Dataset)")
    # byte_level_BPE_train('./dataset/split/mpd/train.pt', './dataset/split/mpd/val.pt',\
    #                         './dataset/tokenizer/title_bpe', 'mpd', vocab_size=1500, limit_alphabet=600)