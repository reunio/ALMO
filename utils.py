import argparse
import re

import numpy as np
import random
import logging
import time
from datetime import timedelta

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import QED
from rdkit.Chem import RDConfig
import os
import sys

from model.mask_model import MaskNetwork

sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

import torch

from rdkit.Chem import MACCSkeys
from libsvm.svmutil import *

import collections


def contains_benzene_ring(smiles, args):
    group_remain = args.keep_benzene_ring
    result = True

    if group_remain == 0:
        return result

    # 苯环
    if group_remain == 1:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return False
        benzene_smarts = Chem.MolFromSmarts('c1ccccc1')
        result = molecule.HasSubstructMatch(benzene_smarts)

    if group_remain == 2:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return False
        hydroxyl_smarts = Chem.MolFromSmarts('[OH]')
        benzene_smarts = Chem.MolFromSmarts('c1ccccc1')
        result = molecule.HasSubstructMatch(hydroxyl_smarts) and molecule.HasSubstructMatch(benzene_smarts)

    # 羧基
    if group_remain == 3:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return False
        carboxyl_smarts = Chem.MolFromSmarts('[CX3](=[OX1])[OX2H1]')
        result = molecule.HasSubstructMatch(carboxyl_smarts)

    return result


def draw_mol_list(smiles_list, label_list, path, show=False):
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    mol_list = mol_list[0::10]
    label_list = label_list[0::10]
    mols_per_row = 5
    sub_img_size = (300, 300)
    img = Draw.MolsToGridImage(mol_list, legends=label_list, molsPerRow=mols_per_row, subImgSize=sub_img_size)
    if show:
        img.show()
    img.save(path)



def restore_smiles(smiles):
    if smiles[-1] != "%":
        smiles += "%"
    start_index = smiles.index('$') + 1
    end_index = smiles.index('%')
    substring = smiles[start_index:end_index]
    smiles = smiles.split('$')[0]
    index = smiles.find('&')
    output_smiles = (smiles[:index] + substring + smiles[index:end_index]).replace('$', "").replace('%', "").replace('&', "")
    return output_smiles

def random_mask_smiles(swarm, args):

    new_swarm = []
    replaced_chars = []

    for i in range(args.swarm_num):
        string = swarm[i]
        length = len(string)

        start = random.randint(0, length - 1)
        end = random.randint(start + 1, min(start + args.max_mask_len, length))

        replaced = string[start:end]
        replaced_chars.append(replaced)
        Q_number = random.randint(1, args.max_mask_len)
        replaced_string = string[:start] + "Q" * Q_number + string[end:]
        if args.only_one_mask:
            replaced_string = re.sub(r'Q+', '&', replaced_string)
        else:
            replaced_string = replaced_string.replace('Q', '&')

        new_swarm.append(replaced_string)

    return new_swarm, replaced_chars

def mask_model_mask_smiles(swarm, args, gpt_m, m, tokenizer):
    new_swarm = []
    replaced_chars = []

    encodings_dict = tokenizer(swarm,
                               truncation=True, max_length=512, padding="max_length")
    input_one = torch.tensor(encodings_dict['input_ids'])

    outputs = gpt_m(input_one)
    outputs = m(outputs)
    sampled_indices = torch.multinomial(outputs, 1)

    for i in range(args.swarm_num):
        string = swarm[i]
        length = len(string)
        print(string)
        start = min(sampled_indices[i] // 100,sampled_indices[i] % 100)
        end = max(sampled_indices[i] // 100,sampled_indices[i] % 100)

        if start > end:
            start = end = length
        if start < length and length < end:
            end = length

        replaced = string[start:end]
        replaced_chars.append(replaced)

        replaced_string = string[:start] + "Q" + string[end:]

        print(replaced_string)

        new_swarm.append(replaced_string)

    return new_swarm, replaced_chars


def load_mask_model(model_path):
    configuration = GPT2Config.from_pretrained(model_path, output_hidden_states=True)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path, pad_token='<|pad|>')

    gpt_model = GPT2LMHeadModel.from_pretrained(model_path, config=configuration)

    weight_model = MaskNetwork()
    weight_model.load_state_dict(torch.load(model_path + '/weight_model.pth', map_location=torch.device('cpu')))

    return gpt_model, weight_model, tokenizer


def set_train_logger(name=''):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_name = './log/train_'+name+'.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger

def set_test_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_name = './log/test.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

def set_ES_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    timestamp = str(int(time.time()))
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_name = './log/ES_test'+timestamp+'.log'
    logfile = log_name
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def record_operating_parameters(logger, args):
    logger.info("epochs")
    logger.info(args.epochs)
    logger.info("seed")
    logger.info(args.seed)
    logger.info("batch_size")
    logger.info(args.batch_size)
    logger.info("sample_size")
    logger.info(args.sample_size)

    logger.info("------------------------------for finetune------------------------------")
    logger.info("fname")
    logger.info(args.data_fname)
    logger.info("sample")
    logger.info(args.sample)

    logger.info("------------------------------for test------------------------------")
    logger.info("fname_test")
    logger.info(args.data_fname_test)
    logger.info("sample_test")
    logger.info(args.sample_test)
    logger.info("test_model_path")
    logger.info(args.test_model_path)
    logger.info("batch_size_test")
    logger.info(args.batch_size_test)
    logger.info("epochs_test")
    logger.info(args.epochs_test)

    logger.info("------------------------------for ES-----------------------------")
    logger.info("swarm_num")
    logger.info(args.swarm_num)
    logger.info("max_mask_len")
    logger.info(args.max_mask_len)
    logger.info("similar_weight")
    logger.info(args.similar_weight)
    logger.info("min_similar")
    logger.info(args.min_similar)
    logger.info("opti_object")
    logger.info(args.opti_object)
    logger.info("allow_individuals_repeat")
    logger.info(args.allow_individuals_repeat)
    logger.info("keep_benzene_ring")
    logger.info(args.keep_benzene_ring)



def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def setup_seed(seed):
    if seed != 0:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


def format_time(elapsed):
    return str(timedelta(seconds=int(round((elapsed)))))



def end_epoch_log(args, logger, his_best_qed, his_best_logp, his_best_drd2, his_best_fitness, his_best_similar):

    # logger.info("his_best_fitness")
    # logger.info(his_best_fitness)
    if args.opti_object == "qed":
        num = 0
        aaa = 0
        bbb = 0
        for i in his_best_qed:
            num = num + 1
            if i[-1] > 0.9:
                aaa = aaa + 1
            if i[-1] - i[0] > 0.1:
                bbb = bbb + 1
        logger.info("qed > 0.9")
        logger.info(aaa / num)
        logger.info("qed improv > 0.1")
        logger.info(bbb / num)

    if args.opti_object == "logp":
        before = []
        after = []
        for i in range(len(his_best_logp)):
            before.append(his_best_logp[i][0])
            after.append(his_best_logp[i][-1])

        logger.info("Average logP before optimization")
        logger.info(sum(before) / len(before))
        logger.info("Average logP after optimization")
        logger.info(sum(after) / len(after))

    if args.opti_object == "drd2":
        before = []
        after = []
        for i in range(len(his_best_drd2)):
            before.append(his_best_drd2[i][0])
            after.append(his_best_drd2[i][-1])

        logger.info("Average drd2 before optimization")
        logger.info(sum(before) / len(before))
        logger.info("Average drd2 after optimization")
        logger.info(sum(after) / len(after))

        num_d = 0
        aaa_q = 0
        aaa_d = 0
        for i in his_best_drd2:
            num_d = num_d + 1
            if i[-1] > 0.5:
                aaa_q = aaa_q + 1
            if i[-1] - i[0] > 0.2:
                aaa_d = aaa_d + 1
        logger.info("drd2 > 0.5")
        logger.info(aaa_q / num_d)
        logger.info("drd2 imporov > 0.2")
        logger.info(aaa_d / num_d)

    if args.opti_object == "drd2qed":
        num_d = 0
        aaa_q = 0
        aaa_d = 0
        aaa_q_d = 0
        for i in his_best_qed:
            num_d = num_d + 1
            if i[-1] > 0.6:
                aaa_q = aaa_q + 1
        for j in his_best_drd2:
            if j[-1] > 0.5:
                aaa_d = aaa_d + 1
        for k in range(len(his_best_qed)):
            if his_best_qed[k][-1]>0.6 and his_best_drd2[k][-1]>0.5:
                aaa_q_d = aaa_q_d+1

        logger.info("qed > 0.6")
        logger.info(aaa_q / num_d)
        logger.info("drd2 > 0.5")
        logger.info(aaa_d / num_d)
        logger.info("qed > 0.6 and drd2 > 0.5")
        logger.info(aaa_q_d / num_d)

    final_average_fitness = [row[-1] for row in his_best_fitness]
    logger.info("final average fitness")
    logger.info(sum(final_average_fitness) / len(final_average_fitness))

    final_average_qed = [row[-1] for row in his_best_qed]
    logger.info("final average qed")
    logger.info(sum(final_average_qed) / len(final_average_qed))

    final_average_logp = [row[-1] for row in his_best_logp]
    logger.info("final average logp")
    logger.info(sum(final_average_logp) / len(final_average_logp))

    final_average_drd2 = [row[-1] for row in his_best_drd2]
    logger.info("final average drd2")
    logger.info(sum(final_average_drd2) / len(final_average_drd2))

    final_average_similar = [row[-1] for row in his_best_similar]
    logger.info("final average similar")
    logger.info(sum(final_average_similar) / len(final_average_similar))

def get_parse():
    parser = argparse.ArgumentParser()

    # pre-parsing args
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--backbone", type=str, default='gpt2', choices=['biobert','bert','gpt2'])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--sample_size", type=int, default=3000000)
    parser.add_argument("--gpu_num", type=int, default=8)

    parser.add_argument('--only_one_mask', action='store_true')


    # for finetune
    parser.add_argument("--data_fname", type=str, default='data/EP2_mask_.csv')
    parser.add_argument("--sample", type=int, default=1900) # 400000
    parser.add_argument("--finetune_model_path", type=str, default='gpt2_small')
    parser.add_argument("--add_KL_loss", type=int, default=0)


    # for test
    parser.add_argument("--data_fname_test", type=str, default='data/pubchem/pubchem_100_900_83_mask_5.csv')

    parser.add_argument("--sample_test", type=int, default=800)  # 400000
    parser.add_argument("--test_model_path", type=str, default='model_save_sample_QED0.9_500000_mask_10_muti_mask_420000_4')
    parser.add_argument("--batch_size_test", type=int, default=2)
    parser.add_argument("--epochs_test", type=int, default=1)

    parser.add_argument("--opti_object", type=str, default='qed')
    parser.add_argument("--swarm_num", type=int, default=5)
    parser.add_argument("--max_mask_len", type=int, default=5)
    parser.add_argument("--similar_weight", type=float, default=0.25)
    parser.add_argument("--min_similar", type=float, default=0.6)

    parser.add_argument("--mask_model_path", type=str, default='mask_model_gpt_save_drd2MMP_50000_1')

    # 是否保持苯环
    parser.add_argument("--keep_benzene_ring", type=int, default=0)


    # for RL
    parser.add_argument("--RL_model_path", type=str, default='model_save_120000_12_new')
    parser.add_argument('--epoch_start', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--RL_epoch_size', type=int, default=2)
    parser.add_argument('--RL_batch_size', type=int, default=1)
    parser.add_argument('--T_train', type=int, default=200)
    parser.add_argument('--n_step', type=int, default=1)
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--gamma', type=float, default=0.8)


    # for evaluation
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--eval_type", type=str, default='n')  # normal/drop/shuffle
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--return_num", type=int, default=5)
    parser.add_argument("--task", type=str, default='pretrain')
    parser.add_argument('--allow_individuals_repeat', action='store_true')

    args = parser.parse_args()

    args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    return args