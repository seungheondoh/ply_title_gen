# PlyTitle_Generation

This is the official repository of **Music Playlist Title Generation: A Machine-Translation Approach**. The paper has been accepted by [2nd Workshop on NLP for Music and Spoken Audio](https://sites.google.com/view/nlp4musa-2021) co-located with ISMIR'2021.

- [Pretrained Weight and Datasplit](https://zenodo.org/record/5804567#.Ycd7kxNBxb8)
- [Presentation](https://www.youtube.com/watch?v=bzg6TO6mcrw&list=PL44xXQ2KNZ0IXb7mZGtHHBQwbPqH5OMvc&index=3&ab_channel=NLP4MusA)


### Inference Results

This model use the track_id sequence as an input and return the track title sequence as an output. 

- Melon dataset's results can be found [here]().

```
    "25269": {
        "ground_truth": "취향저격 감성힙합+α 두번째",
        "prediction": "r&b soul introduction 버벌진트 <eos>"
    },
    "69588": {
        "ground_truth": "걸그룹의 대표적인 곡들",
        "prediction": "신나는 댄스곡 모음 <eos>"
    },
    "66941": {
        "ground_truth": "미리 메리 크리스마스",
        "prediction": "크리스마스 캐롤 크리스마스 캐롤 <eos>"
    },
```

- Spotify-million-playlist-dataset dataset's results can be found [here]().

```
    "923035": {
        "ground_truth": "wedding dinner music",
        "prediction": "wedding - cocktail hour <eos>"
    },
    "634077": {
        "ground_truth": "history of rap",
        "prediction": "old school hip hop <eos>"
    }
    "540451": {
        "ground_truth": "metal up your ass",
        "prediction": "rock and roll <eos>"
    },
```

### Environment

1. Install python and PyTorch:
    - python==3.8.5
    - torch==1.8.0 (Please install it according to your [CUDA version](https://pytorch.org/get-started/previous-versions/#linux-and-windows-4).)
    
2. Other requirements:
    - pip install -r requirements.txt

### Training from scratch
1. Download the data files from [spotify-million-playlist](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/dataset_files) and [Melon Kakao](https://arena.kakao.com/melon_dataset).

2. Run preprocessing code and split dataset

```
python preprocessing.py
```
or you can download pre-split dataset from [here](https://zenodo.org/record/5804567#.Ycd7kxNBxb8).

3. Training options (best pramas):  

```
python train.py --dataset_type melon --model transfomer --shuffle True --e_pos False
python train.py --dataset_type mpd --model transfomer --shuffle True --e_pos False
```

4. Evaluation & Inference

```
python eval.py --dataset_type melon --model transfomer --shuffle True --e_pos False
python infer.py --dataset_type melon --model transfomer --shuffle True --e_pos False
```

### Reference 

https://github.com/bentrevett/pytorch-seq2seq