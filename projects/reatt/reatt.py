"""
Trains a a modified minGPT model with 
    * rotary positional encodings,
    * SwiGLU,
    * a learning rate warmup,
    * a cosine LR scheduler, and
    * RMSNorm.
"""

import os
import sys

import torch
from torch.utils.data import Dataset
# from torch.utils.data.dataloader import DataLoader
from mingpt.jsonl_dataset import JSONLDataset
from mingpt.jsonl_dataset import tokenizer  # reuse the already-loaded GPT-2 tokenizer

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN

# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # system
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/reatt'

    # data
    C.data = CN()
    C.data.file_path = '../pile_data_10.jsonl'
    C.data.split = 'train'
    C.data.test_size = 10
    C.data.block_size = 256

    # model
    C.model = GPT.get_default_config()
    C.model.model_type = 'gpt-mini'

    # trainer
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
    C.trainer.log_interval = 100
    C.trainer.ckpt_interval = 500

    return C

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    # get default config and overrides from the command line, if any
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    print(config)
    setup_logging(config)
    set_seed(config.system.seed)

    # construct the training dataset
    train_dataset = JSONLDataset(config.data)

    # construct the model
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    model = GPT(config.model)

    # construct the trainer object
    trainer = Trainer(config.trainer, model, train_dataset)

    
    log_path = os.path.join(config.system.work_dir, "loss.csv")
    def batch_end_callback(trainer):
        if trainer.iter_num % config.trainer.log_interval == 0:
            # write header once
            if trainer.iter_num == 0 and not os.path.exists(log_path):
                with open(log_path, "w") as f:
                    f.write("iter,loss,dt\n")
            with open(log_path, "a") as f:
                f.write(f"{trainer.iter_num},{trainer.loss.item()},{trainer.iter_dt}\n")

        if trainer.iter_num % 500 == 0:
            model.eval()
            with torch.no_grad():
                prompt = "When treating mice" # I put in this prompt
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                x = torch.tensor(ids, dtype=torch.long)[None, ...].to(trainer.device)
                y = model.generate(x, 100, temperature=1.0, do_sample=True, top_k=10)[0].tolist()
                print(tokenizer.decode(y, clean_up_tokenization_spaces=True))
            model.train()

        # if trainer.iter_num % config.trainer.ckpt_interval == 0 and trainer.iter_num > 0:
        #     ckpt_path = os.path.join(config.system.work_dir, f"model_iter_{trainer.iter_num}.pt")
        #     torch.save(model.state_dict(), ckpt_path)
        #     # optionally also save latest
        #     torch.save(model.state_dict(), os.path.join(config.system.work_dir, "model.pt"))

    trainer.set_callback('on_batch_end', batch_end_callback)
    
    # run the optimization
    trainer.run()
