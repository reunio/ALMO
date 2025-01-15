import time

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel

from DRD2_SVM import drd2_model
from utils import get_parse, setup_seed, set_train_logger, format_time, set_test_logger, restore_smiles, \
    record_operating_parameters, random_mask_smiles, set_ES_logger, draw_mol_list, contains_benzene_ring
from load_data import load_train_data, load_test_data, load_ES_data
from accelerate import Accelerator
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, rdMolDescriptors, AllChem, Descriptors
from tqdm import tqdm, trange

# set logger
logger = set_ES_logger()

logger.info("4ES_test.py")

args = get_parse()
record_operating_parameters(logger, args)

accelerator = Accelerator()
device = accelerator.device
print(device)

setup_seed(args.seed)
# Load the GPT tokenizer.
tokenizer = GPT2Tokenizer.from_pretrained(args.test_model_path,
                                          pad_token='<|pad|>')
pad_token_id = tokenizer.pad_token_id
eos_token_id = tokenizer.encode("%", add_special_tokens=False)[0]

configuration = GPT2Config.from_pretrained(args.test_model_path, output_hidden_states=False)

model = GPT2LMHeadModel.from_pretrained(args.test_model_path, config=configuration)

model.resize_token_embeddings(len(tokenizer))
model.to(device)

model.eval()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(DEVICE)

drd2_svm = drd2_model()

logger.info("loading test data")
test_data, test_data_size = load_ES_data(tokenizer, args)

logger.info("loading test end")

epochs = args.epochs_test
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8
swarm_num = args.swarm_num

batch_num = int(args.sample_size / args.gpu_num / args.batch_size_test) + 1

total_t0 = time.time()

swarms = []
swarms_fitness = []
swarms_qed = []
now_best = []
now_best_fitness = []
now_best_qed = []
now_best_similar = []
his_best = []
his_best_fitness = []
his_best_similar = []
his_best_qed = []

for step, one_sm in tqdm(enumerate(test_data.smiles), total=len(test_data.smiles), desc="init..."):
    swarms.append([one_sm] * swarm_num)

    molecule = Chem.MolFromSmiles(one_sm)

    one_sm_qed = 0
    if args.opti_object == "qed":
        one_sm_qed = QED.qed(molecule)
    elif args.opti_object == "logp":
        one_sm_qed = Descriptors.MolLogP(molecule)
    else:
        print("illegal  opti_object")
        exit()

    swarms_fitness.append([one_sm_qed] * swarm_num)
    swarms_qed.append([one_sm_qed] * swarm_num)

    now_best_fitness.append(one_sm_qed)
    now_best.append(one_sm)
    now_best_qed.append(one_sm_qed)
    now_best_similar.append(0)

    his_best.append([one_sm])
    his_best_fitness.append([one_sm_qed])
    his_best_qed.append([one_sm_qed])
    his_best_similar.append([1])

logger.info("init mol list")
logger.info(now_best)
logger.info(now_best_fitness)
logger.info(now_best_qed)
logger.info(sum(now_best_qed) / len(now_best_qed))

for epoch_i in trange(0, epochs):

    if accelerator.is_main_process:
        logger.info("")
        logger.info('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        logger.info('SIA Testing...')

    remaining_epoch = args.epochs_test - epoch_i - 1
    passed_batch = 0

    passed_batch = 0

    valid_num = 0

    for step, one_swarm in enumerate(swarms):

        t0 = time.time()

        one_swarm_qed = swarms_qed[step]

        one_swarm_mask, _ = random_mask_smiles(one_swarm, args)

        new_one_swarm = []
        new_one_swarm_qed = []

        for i in range(args.swarm_num):

            one_smiles = one_swarm_mask[i]
            one_smiles_value = swarms_fitness[step]

            encodings_dict = tokenizer(one_smiles + '$')
            input_one = torch.tensor(encodings_dict['input_ids'])
            b_masks_one = torch.tensor(encodings_dict['attention_mask'])

            input_text = input_one.unsqueeze(0).to(device)
            b_masks = b_masks_one.unsqueeze(0).to(device)
            max_length = 512

            with torch.no_grad():
                outputs = model.generate(input_text,
                                         attention_mask=b_masks,
                                         max_length=max_length,
                                         num_return_sequences=1,
                                         do_sample=True,
                                         pad_token_id=pad_token_id,
                                         eos_token_id=eos_token_id,
                                         )

            generated_text = tokenizer.decode(outputs[0]).replace(' <|pad|>', '').replace(' ', '')

            smiles = restore_smiles(generated_text)

            try:
                # 修改后的
                molecule = Chem.MolFromSmiles(smiles)

                qed = 0
                if args.opti_object == "qed":
                    qed = QED.qed(molecule)
                elif args.opti_object == "logp":
                    qed = Descriptors.MolLogP(molecule)

                new_one_swarm_qed.append(qed)
                new_one_swarm.append(smiles)

                valid_num = valid_num + 1

            except:
                new_one_swarm.append(one_swarm[i])
                new_one_swarm_qed.append(one_swarm_qed[i])

        no_repeat_swarm = []
        no_repeat_swarm_qed = []
        for qq in range(len(new_one_swarm)):
            if new_one_swarm[qq] not in one_swarm:
                no_repeat_swarm.append(new_one_swarm[qq])
                no_repeat_swarm_qed.append(new_one_swarm_qed[qq])

        temp_swarm = one_swarm + no_repeat_swarm
        temp_swarm_qed = one_swarm_qed + no_repeat_swarm_qed

        temp_similar = []
        lead_mol = his_best[step][0]
        for sim_i in temp_swarm:

            x_1 = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(lead_mol), 2)

            try:
                x_2 = AllChem.GetMorganFingerprint(Chem.MolFromSmiles(sim_i), 2)
                similarity = DataStructs.TanimotoSimilarity(x_1, x_2)

                if contains_benzene_ring(sim_i, args):
                    temp_similar.append(similarity)
                else:
                    temp_similar.append(0)

            except Exception:
                temp_similar.append(0)

        # 综合计算适应度
        temp_swarm_fitness = []
        for k in range(len(temp_swarm_qed)):

            if temp_similar[k] <= args.min_similar:
                temp_swarm_fitness.append(0)
            else:
                temp_swarm_fitness.append(temp_swarm_qed[k])

        molecular_data = list(zip(temp_swarm, temp_swarm_fitness, temp_swarm_qed, temp_similar))

        sorted_data = sorted(molecular_data, key=lambda x: x[1], reverse=True)

        top_5_swarm = [x[0] for x in sorted_data[:swarm_num]]
        top_5_swarm_fitness = [x[1] for x in sorted_data[:swarm_num]]
        top_5_swarm_qed = [x[2] for x in sorted_data[:swarm_num]]
        top_5_swarm_similar = [x[3] for x in sorted_data[:swarm_num]]

        max_swarm_fitness = max(top_5_swarm_fitness)
        if max_swarm_fitness > now_best_fitness[step]:
            max_index = top_5_swarm_fitness.index(max_swarm_fitness)
            now_best_fitness[step] = max_swarm_fitness
            now_best[step] = top_5_swarm[max_index]
            now_best_qed[step] = top_5_swarm_qed[max_index]
            now_best_similar[step] = top_5_swarm_similar[max_index]

        swarms[step] = top_5_swarm
        swarms_fitness[step] = top_5_swarm_fitness
        swarms_qed[step] = top_5_swarm_qed

        his_best[step].append(now_best[step])
        his_best_fitness[step].append(now_best_fitness[step])
        his_best_qed[step].append(now_best_qed[step])
        his_best_similar[step].append(now_best_similar[step])

        passed_batch = passed_batch + 1
        time_elapse = time.time() - t0

    training_time = format_time(time_elapse)

    # remaining batches and time
    remaining_batches = batch_num - passed_batch
    remaining_time = format_time(remaining_batches * time_elapse)

    if accelerator.is_main_process:
        logger.info("\n")
        logger.info(f"Current at the {epoch_i}-th epoch.")
        logger.info("  Trained batch: {}".format(passed_batch))
        logger.info("  Remaining batches: {}".format(remaining_batches))
        logger.info("  Time need: {:}.".format(remaining_time))
        logger.info("Valid_rate")
        logger.info(valid_num / (test_data_size * args.swarm_num))
        logger.info("new_best_fitness_list")
        logger.info(now_best_fitness)
        logger.info("new_best_list")
        logger.info(now_best)


logger.info("\n")

logger.info("his_best")
logger.info(his_best)

logger.info("his_best_fitness")
if args.opti_object == "qed":
    num = 0
    aaa = 0
    bbb = 0
    for i in his_best_fitness:
        num = num + 1
        if i[-1]>0.9:
            aaa = aaa + 1
        if i[-1] - i[0] > 0.1:
            bbb = bbb + 1
    logger.info("qed大于0.9")
    logger.info(aaa/num)
    logger.info("qed增长大于0.1")
    logger.info(bbb/num)

if args.opti_object == "logp":
    before = []
    after = []
    for i in range(len(his_best_fitness)):
        before.append(his_best_fitness[i][0])
        after.append(his_best_fitness[i][-1])

    logger.info("优化前的平均logp")
    logger.info(sum(before)/len(before))
    logger.info("优化后的平均logp")
    logger.info(sum(after)/len(after))


final_average_fitness = [row[-1] for row in his_best_fitness]
logger.info("final average fitness")
logger.info(sum(final_average_fitness) / len(final_average_fitness))

final_average_qed = [row[-1] for row in his_best_qed]
logger.info("final average qed")
logger.info(sum(final_average_qed) / len(final_average_qed))

final_average_similar = [row[-1] for row in his_best_similar]
logger.info("final average similar")
logger.info(sum(final_average_similar) / len(final_average_similar))

logger.info("\n")
logger.info("Testing complete!")
logger.info("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))
