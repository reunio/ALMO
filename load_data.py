import re

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class GPT2Dataset(Dataset):
    def __init__(self, data_df, tokenizer, args, max_length=512):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        self.smiles = []

        for index, data in tqdm(data_df.iterrows(), total=len(data_df), desc="Loading data"):
            mask_smiles = data['smiles_Q']
            tg = data['target']
            tg_smiles = data['sd_smiles_Q']

            if args.only_one_mask:
                mask_smiles = re.sub(r'Q+', '&', mask_smiles)
            else:
                mask_smiles = mask_smiles.replace('Q', '&')

            # $3 %4 &5
            encodings_dict = tokenizer(mask_smiles + '$' + tg + '%',
                                                        truncation=True, max_length=max_length, padding="max_length")
            input_one = torch.tensor(encodings_dict['input_ids'])
            self.input_ids.append(input_one)

            input_one_clone = input_one.clone()

            masks_one = torch.tensor(encodings_dict['attention_mask'])
            self.attn_masks.append(masks_one)

            def modify_tensor_2(ts):
                labels = ts[1:]
                new_element = torch.tensor([50257], dtype=torch.int32)
                tensor_label = torch.cat((labels, new_element), dim=0)
                return tensor_label

            self.labels.append(modify_tensor_2(input_one_clone))

            pass

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx], self.attn_masks[idx]

class GPT2Dataset_wr(Dataset):
    def __init__(self, data_df, tokenizer, args, max_length=512):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        self.smiles = []

        for index, data in tqdm(data_df.iterrows(), total=len(data_df), desc="Loading data"):
            mask_smiles = data['smiles_Q']
            tg = data['target']
            tg_smiles = data['sd_smiles_Q']

            if args.only_one_mask:
                mask_smiles = re.sub(r'Q+', '&', mask_smiles)
            else:
                mask_smiles = mask_smiles.replace('Q', '&')

            # $3 %4 &5
            encodings_dict = tokenizer(mask_smiles + '$' + tg + '%',
                                                        truncation=True, max_length=max_length, padding="max_length")
            input_one = torch.tensor(encodings_dict['input_ids'])
            self.input_ids.append(input_one)

            input_one_clone = input_one.clone()

            masks_one = torch.tensor(encodings_dict['attention_mask'])
            self.attn_masks.append(masks_one)

            def modify_tensor_2(ts):
                labels = ts[1:]
                new_element = torch.tensor([50257], dtype=torch.int32)
                tensor_label = torch.cat((labels, new_element), dim=0)
                return tensor_label

            self.labels.append(modify_tensor_2(input_one_clone))

            pass

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx], self.attn_masks[idx]

class GPT2Dataset_mask(Dataset):
    def __init__(self, data_df, tokenizer, args, max_length=512):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.labels = []
        self.smiles = []
        self.site_tensor = []

        for index, data in tqdm(data_df.iterrows(), total=len(data_df), desc="Loading data"):
            mask_smiles = data['sd_smiles_Q']
            mask_site_bg = data['mask_site_bg']
            mask_site_ed = data['mask_site_ed']

            # $3 %4 &5
            encodings_dict = tokenizer(mask_smiles, truncation=True, max_length=max_length, padding="max_length")
            input_one = torch.tensor(encodings_dict['input_ids'])
            self.input_ids.append(input_one)

            input_one_clone = input_one.clone()

            masks_one = torch.tensor(encodings_dict['attention_mask'])
            self.attn_masks.append(masks_one)

            def modify_tensor_2(ts):
                labels = ts[1:]
                new_element = torch.tensor([50257], dtype=torch.int32)  # 创建新元素
                tensor_label = torch.cat((labels, new_element), dim=0)  # 在末尾拼接新元素
                return tensor_label

            self.labels.append(modify_tensor_2(input_one_clone))

            if mask_site_bg < 99:
                if mask_site_ed < 99:
                    pass
                else:
                    mask_site_ed = 99
            else:
                mask_site_bg = 99
                mask_site_ed = 99

            aaa = torch.zeros(10000)
            aaa[mask_site_bg*100+mask_site_ed] = 1

            self.site_tensor.append(aaa)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.labels[idx], self.attn_masks[idx], self.site_tensor[idx]

class GPT2Dataset_test(Dataset):
    def __init__(self, data_df, tokenizer, args, max_length=512):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.smiles = []
        self.attn_masks = []

        self.smiles_Q = []
        self.sd_smiles_Q = []
        self.target = []

        for index, data in tqdm(data_df.iterrows(), total=len(data_df), desc="Loading data"):
            self.smiles.append(data['sd_smiles_Q'])
            mask_smiles = data['smiles_Q']
            tg_smiles = data['sd_smiles_Q']

            self.smiles_Q.append(mask_smiles)
            self.sd_smiles_Q.append(tg_smiles)
            self.target.append(data['target'])

            # 现在设置
            if args.only_one_mask:
                mask_smiles = re.sub(r'Q+', '&', mask_smiles)
            else:
                mask_smiles = mask_smiles.replace('Q', '&')

            encodings_dict = tokenizer(mask_smiles + '$')
            input_one = torch.tensor(encodings_dict['input_ids'])

            self.input_ids.append(input_one)

            masks_one = torch.tensor(encodings_dict['attention_mask'])
            self.attn_masks.append(masks_one)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

    def get_origin_smiles(self):
        return self.smiles


class GPT2Dataset_ES(Dataset):
    def __init__(self, data_df, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.smiles = []

        for index, data in tqdm(data_df.iterrows(), total=len(data_df), desc="Loading data"):
            self.smiles.append(data['sd_smiles_Q'])

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx]


def get_chunks():
    chunks = []

    file_1 = './data/train/pretrain_8000k.csv'
    df_1 = pd.read_csv(file_1, iterator=True)

    chunks.append(df_1)

    return chunks

def fetch(args, tokenizer, chunk):
    fetch_size = args.batch_size * args.gpu_num * 100 if args.sample_size > 50000 else args.sample_size
    try:
        data_df = chunk.get_chunk(fetch_size)
        fetched_size = len(data_df)
    except Exception as e:
        return None, 0

    train_data = GPT2Dataset(data_df, tokenizer, args)

    return train_data, fetched_size

def load_train_data(tokenizer, args):
    fname = args.data_fname
    data_df = pd.read_csv(fname)
    data_df = data_df.sample(n=args.sample)
    train_data = GPT2Dataset(data_df, tokenizer, args)
    train_data_size = len(data_df)
    return train_data, train_data_size

def load_train_mask_data(tokenizer, args):
    fname = args.data_fname
    data_df = pd.read_csv(fname)
    data_df = data_df.sample(n=args.sample)
    train_data = GPT2Dataset_mask(data_df, tokenizer, args)
    train_data_size = len(data_df)
    return train_data, train_data_size

def load_test_data(tokenizer, args):
    fname = args.data_fname_test
    data_df = pd.read_csv(fname)
    data_df = data_df.sample(n=args.sample_test)
    train_data = GPT2Dataset_test(data_df, tokenizer, args)
    train_data_size = len(data_df)
    return train_data, train_data_size

def load_RL_data(tokenizer, args):
    fname = args.data_fname_RL
    data_df = pd.read_csv(fname)
    data_df = data_df.sample(n=args.sample)
    train_data = GPT2Dataset(data_df, tokenizer, args)
    train_data_size = len(data_df)
    return train_data, train_data_size

def load_ES_data(tokenizer, args):
    fname = args.data_fname_test
    data_df = pd.read_csv(fname)
    data_df = data_df.sample(n=args.sample_test)
    train_data = GPT2Dataset_ES(data_df, tokenizer)
    train_data_size = len(data_df)
    return train_data, train_data_size

if __name__ == '__main__':
    pass