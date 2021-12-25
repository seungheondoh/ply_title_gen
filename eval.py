from argparse import ArgumentParser, Namespace
from pathlib import Path
import json
import os
import torch
import pickle
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from src.model.rnn import RNN_Attn
from src.model.transfomer import Transfomer
from src.task.pipeline import PlyPipeline
from src.task.runner import Runner, TF_Runner

from tokenizers import Tokenizer, SentencePieceBPETokenizer
from src.utils import Vocab
import wandb

class _tokenizer():
    def __init__(self, dataset_type):
        self.tokenizer_dir = './dataset/tokenizer/title_bpe'
        self.dataset_type = dataset_type
        self.model = SentencePieceBPETokenizer(os.path.join(self.tokenizer_dir, "{}-vocab.json".format(self.dataset_type)),\
                                            os.path.join(self.tokenizer_dir, "{}-merges.txt".format(self.dataset_type)))
        
        self.encoder = self.model.get_vocab()
    def encode(self, target_str, is_pretokenized=False, add_special_tokens=True):
        return self.model.encode(target_str, pair=None, is_pretokenized=False, add_special_tokens=True).ids

    def decode(self, target_ids, skip_special_tokens=True):
        return self.model.decode(target_ids, skip_special_tokens=True)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_tensorboard_logger(args: Namespace) -> TensorBoardLogger:
    logger = TensorBoardLogger(
        save_dir=f"exp/{args.dataset_type}", name=f"{args.model}", version=f"{args.tokenzier}/e:{args.embed_size}_h:{args.hidden_size}_el:{args.e_layers}_dl:{args.d_layers}_tf:{args.teacher_forcing_ratio}_out:{args.dropout}_s:{args.shuffle}"
    )
    return logger

def get_wandb_logger(model):
    logger = WandbLogger()
    logger.watch(model)
    return logger 

def get_checkpoint_callback(args, save_path) -> ModelCheckpoint:
    prefix = save_path
    suffix = "best"
    checkpoint_callback = ModelCheckpoint(
        dirpath=prefix,
        filename=suffix,
        save_top_k=1,
        save_last= False,
        monitor="val_loss",
        mode='min',
        save_weights_only=True,
        verbose=True,
    )
    return checkpoint_callback

def get_early_stop_callback(args: Namespace) -> EarlyStopping:
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=0.00, patience=20, verbose=True, mode="min"
    )
    return early_stop_callback


def main(args) -> None:
    if args.reproduce:
        seed_everything(42)

    config = OmegaConf.create()
    config.update(hparams=vars(args))

    wandb.init(config=args)
    args = wandb.config
    save_path = f"exp/{args.dataset_type}/{args.model}/{args.tokenzier}/s:{args.shuffle}_epos:{args.e_pos}"
    title_tokenizer = _tokenizer(dataset_type = args.dataset_type)
    song_vocab = pickle.load(open(os.path.join("./dataset/tokenizer/track", args.dataset_type + "_vocab.pkl"), mode="rb"))
    title_vocab = pickle.load(open(os.path.join("./dataset/tokenizer/title_split", args.dataset_type + "_vocab.pkl"), mode="rb"))
    pipeline = PlyPipeline(
                split_path=args.split_path,
                tokenzier = args.tokenzier,
                dataset_type=args.dataset_type,
                context_length=args.context_length,
                title_tokenizer= title_tokenizer,
                title_vocab = title_vocab,
                song_vocab= song_vocab,
                shuffle = args.shuffle,
                batch_size=args.batch_size,
                num_workers=args.num_workers
                )
    if args.tokenzier == "white":
        input_size = len(song_vocab) 
        output_size= len(title_vocab)
    else:
        raise ValueError("Current model only support white space tokenizer")

    if args.model == "rnn":
        model = RNN_Attn(
                    input_size = input_size, 
                    output_size= output_size,
                    embed_size= args.embed_size,
                    hidden_size= args.hidden_size,
                    e_layers= args.e_layers, 
                    d_layers= args.d_layers, 
                    dropout= args.dropout, 
                    teacher_forcing_ratio = args.teacher_forcing_ratio
                    )
        runner = Runner(model=model, 
                    lr = args.lr, 
                    weight_decay = args.weight_decay, 
                    T_0 = args.T_0,
                    vocab_size= output_size,
                    )

    elif args.model == "transfomer":
        model = Transfomer(
            input_size = input_size, 
            output_size= output_size,
            hidden_size= args.embed_size,
            e_layers= args.e_layers, 
            d_layers= args.d_layers, 
            heads = args.heads,
            pf_dim = args.hidden_size,
            dropout= args.dropout, 
            e_pos = args.e_pos,
            device = args.gpus
        )
        runner = TF_Runner(model=model, 
                    lr = args.lr, 
                    weight_decay = args.weight_decay, 
                    T_0 = args.T_0,
                    vocab_size= output_size,
                    )

    state_dict = torch.load(os.path.join(save_path, "best.ckpt"))
    runner.load_state_dict(state_dict.get("state_dict"))

    trainer = Trainer(
                        max_epochs= args.max_epochs,
                        gpus= [args.gpus],
                        distributed_backend= args.distributed_backend,
                        benchmark= args.benchmark,
                        deterministic= args.deterministic
                      )

    trainer.test(runner, datamodule=pipeline)

    with open(Path(save_path, "results.json"), mode="w") as io:
        json.dump(runner.test_results, io, indent=4)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--split_path", default="./dataset/split", type=str)
    parser.add_argument("--tid", default="0", type=str)
    parser.add_argument("--model", default="transfomer", type=str)
    parser.add_argument("--tokenzier", default="white", type=str)
    parser.add_argument("--dataset_type", default="melon", type=str)    
    parser.add_argument("--context_length", default=64, type=int)
    parser.add_argument("--shuffle", default=True, type=str2bool)
    # model
    parser.add_argument("--embed_size", default=128, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--e_layers", default=3, type=int)
    parser.add_argument("--d_layers", default=3, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)

    parser.add_argument("--e_pos", default=False, type=str2bool)
    parser.add_argument("--d_pos", default=True, type=str2bool)
    parser.add_argument("--heads", default=8, type=int)
    parser.add_argument("--pf_dim", default=256, type=int)

    parser.add_argument("--teacher_forcing_ratio", default=0.5, type=float)
    # pipeline
    parser.add_argument("--batch_size", default=64, type=float)
    parser.add_argument("--num_workers", default=8, type=float)
    # runner
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--T_0", default=200, type=int)
    parser.add_argument("--max_epochs", default=200, type=int)
    parser.add_argument("--gpus", default=1, type=int)
    parser.add_argument("--distributed_backend", default="dp", type=str)
    parser.add_argument("--deterministic", default=True, type=str2bool)
    parser.add_argument("--benchmark", default=False, type=str2bool)
    parser.add_argument("--reproduce", default=True, type=str2bool)

    args = parser.parse_args()
    main(args)
 