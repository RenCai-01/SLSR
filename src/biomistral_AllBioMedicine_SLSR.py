import os



import os
import time
from datetime import datetime
import torch
import pickle
import argparse
import json
from biomedicinecategories import subcategories, categories

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
import pandas as pd
import numpy as np
np.random.seed(42)

from tqdm import tqdm
from transformers import LlamaTokenizerFast
from transformers import AutoModelForCausalLM
from transformers import AutoConfig  
from laser.LaserWrapper import LaserWrapper
from study_utils.log_utils import Logger
from study_utils.metric_utils import Metrics, DatasetMetrics
from study_utils.time_utils import elapsed_from_str, Progress


import torch
import torch.nn as nn
from transformers import LlamaConfig, LlamaForCausalLM,AutoTokenizer,AutoModelForCausalLM,MistralForCausalLM

from copy import deepcopy



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

choices = ["A", "B", "C", "D"]


def format_subject(subject):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s





def format_example(df, idx, include_answer=True):

    # prompt = df.iloc[idx, 0] + "\n"
    prompt = df.iloc[idx, 0]

    options_count = len(df.columns) - 2
    for j in range(options_count):
        option_label = chr(65 + j)  #
        prompt += "\n{}. {}".format(option_label, df.iloc[idx, j + 1])
    prompt += "\nAnswer:"
    if include_answer:
        prompt += " {}\n\n".format(df.iloc[idx, options_count + 1])
    return prompt

def gen_prompt(train_df, subject, k=-1):
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(
        format_subject(subject)
    )
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        prompt += format_example(train_df, i)
    return prompt


@torch.no_grad()

def eval(args, subject, model, tokenizer, dev_df, test_df): 
    cors = []
    all_probs = []


    answers = [chr(65 + i) for i in range(test_df.shape[1] - 2)]  # A, B, C... 根据选项数量生成

    for i in range(test_df.shape[0]):
        k = args.ntrain
        prompt_end = format_example(test_df, i, include_answer=False)
        train_prompt = gen_prompt(dev_df, subject, k)
        prompt = train_prompt + prompt_end
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        while input_ids.shape[-1] > 2048:
            k -= 1
            train_prompt = gen_prompt(dev_df, subject, k)
            prompt = train_prompt + prompt_end
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        label = test_df.iloc[i, test_df.shape[1] - 1]

        logits = model(
            input_ids=input_ids,
        ).logits[:, -1].flatten()

        logits = logits.to(torch.float32) 


        probs_list = [
            logits[tokenizer(option).input_ids[-1]]
            for option in answers
        ]
        probs_tensor = torch.tensor(probs_list)
        probs = (
            torch.nn.functional.softmax(probs_tensor, dim=0)
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
        )
        pred = answers[np.argmax(probs)]

        cor = pred == label
        cors.append(cor)
        all_probs.append(probs)

    acc = np.mean(cors)
    cors = np.array(cors)
    all_probs = np.array(all_probs)
    print("accuracy {:.3f} - {}".format(acc, subject))

    return cors, acc, all_probs


class MistralExperiment:
    def __init__(self, save_dir, logger):
        self.save_dir = save_dir
        self.logger = logger
        # Object to measure progress (as in time taken and time left to complete)
        self.progress = Progress(logger=logger)
        # Object to compute metrics. We set whether we should consider whitespace and lowercase when evaluating
        self.case_sensitive = False
        self.strip = True
        self.metrics = Metrics(case_sensitive=self.case_sensitive, strip=self.strip)
        # Object to aggregate performance over a dataset
        self.dataset_metric = DatasetMetrics(logger=logger)
        # Device for the experiment
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def evalmodel(self,model,tokenizer,dataset,args):
        dataset_size = len(dataset)
        self.logger.log(f"Starting a new intervention with rate {args.rate}. "
                        f"Dataset size {dataset_size}. Batch size {args.batch_size}")
        time_edit_start = time.time()
        model.to(self.device)
        self.logger.log(f"Edited and put model on {model.device} in time {elapsed_from_str(time_edit_start)}")


        correct_predictions = 0
        total_questions = 0

        # Reset progress timestamp
        self.progress.start()

        for i in tqdm(range(0, dataset_size)):
            if (i - 1) % 100 == 0 and i > 1:

                accuracy = correct_predictions / total_questions if total_questions > 0 else 0.0
                self.logger.log(f"Partial accuracy after {i} examples: {accuracy:.3f}")
                self.progress.print(ex_done=i, ex_left=(dataset_size - i))

            question, options, correct_option, option_mapping = dataset[i]
            inputs = tokenizer(question, return_tensors="pt").to(self.device)

            with torch.no_grad():
                # Generate logits for the question
                logits = model(input_ids=inputs.input_ids).logits[:, -1, :]

                option_probs = []
                for option in options:
                    option_token_id = tokenizer(option, return_tensors="pt").input_ids[0][-1].item()
                    option_prob = torch.nn.functional.softmax(logits, dim=-1)[0, option_token_id].item()
                    option_probs.append(option_prob)

                # Select the option with the highest probability
                predicted_option_idx = np.argmax(option_probs)
                predicted_option_content = options[predicted_option_idx]


                predicted_option_letter = option_mapping.get(predicted_option_content, "Unknown")


                is_correct = predicted_option_letter == correct_option


                correct_predictions += int(is_correct)
                total_questions += 1

                if i % 10 == 0:
                    self.logger.log(
                        f"Question: {question}\n"
                        f"Options: {options}\n"
                        f"Correct Option: {correct_option}\n"
                        f"Predicted Option: {predicted_option_content} ({predicted_option_letter})\n"
                        f"Is Correct: {is_correct}"
                    )

        # Calculate final accuracy
        final_accuracy = correct_predictions / total_questions if total_questions > 0 else 0.0
        self.logger.log(f"Final accuracy on {args.split} set: {final_accuracy:.3f}")



    def intervene(self, model, tokenizer, args, llm_name, layer_indices, rate_for_layer_indices):

        time_edit_start = time.time()

        U = []
        S = []
        Vt = []

        for layernum in range(len(layer_indices)):  # 假设共有32层
            if rate_for_layer_indices[layernum] is not None:  # rate为None表示这个层不需要改变
                model_edit, u, s, vt = LaserWrapper.get_edited_model(
                    model=model,
                    lname=args.lname,
                    lnum=layernum,
                    rate=rate_for_layer_indices[layernum],
                    intervention=args.intervention,
                    logger=self.logger,
                    in_place=True
                )
                self.logger.log(f"Experiment Completed for layer {layernum}.")
            else:
                u = None
                s = None
                vt = None

            # 存储每层的U, S, Vt
            U.append(u)
            S.append(s)
            Vt.append(vt)

        model_edit.to(self.device)
        self.logger.log(f"Edited and put model on {model_edit.device} in time {elapsed_from_str(time_edit_start)}")
        return model_edit, U, S, Vt

    def terminate_and_save(self, predictions, llm_name, args, layer_indices, rate_for_layer_indices):
        self.logger.log("Saving results. Final Performance is given below:")
        self.dataset_metric.terminate()
        self.dataset_metric.print()
        time_start = time.time()

        # Save predictions
        save_pred_fname = f"{self.save_dir}/{llm_name}-predictions-{args.rate}-{args.dtpts}-{args.lnum}.p"
        with open(save_pred_fname, "wb") as f:
            pickle.dump(predictions, f)

        # Save the summary
        save_summary_fname = f"{self.save_dir}/{llm_name}-result-summary-{args.rate}-{args.dtpts}-{args.lnum}.pkl"
        results = self.dataset_metric.agg_to_dict()
        for k, v in args.__dict__.items():
            results["args/%s" % k] = v
        with open(save_summary_fname, "wb") as f:
            pickle.dump(results, f)

        # Print final numbers and return
        self.logger.log(f"Time taken to store all results {elapsed_from_str(time_start)}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def createFile(dirPath):
    if(os.path.exists(dirPath)):
        pass
        print("目录"+dirPath+"已经存在")
    else:
        os.mkdir(dirPath)
        print("创建目录"+dirPath)






if __name__ == '__main__':

    # Step 1: Command line argument
    parser = argparse.ArgumentParser(description='Process Arguments for experiments with LLAMA 2 LLM on CounterFact')

    parser.add_argument('--rate', type=float, default=1, help='rates for intervention')
    parser.add_argument('--dtpts', type=int, default=22000, help='# samples per instruction')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size for evaluation')
    parser.add_argument('--max_len', type=int, default=10, help='maximum length for generation')
    parser.add_argument('--k', type=int, default=10, help='top k for evaluation')
    parser.add_argument('--intervention', type=str, default="rank-reduction",
                        choices=['dropout', 'rank-reduction', 'zero'], help="what type of intervention to perform")
    parser.add_argument('--lname', type=str, default="None",
                        choices=['k_proj', 'q_proj', 'v_proj', 'out_proj', 'fc_in', 'fc_up', 'fc_out',
                                 'None', 'dont', 'all', 'mlp', 'attn'],
                        help="provided which type of parameters to effect")
    parser.add_argument('--lnum', type=int, default=28, help='Layers to edit', choices=list(range(-1, 33)))

    parser.add_argument('--model_path',
                        type=str,
                        default="/yourpath",
                        help="Place where model weights are stored")  #



    parser.add_argument('--lasermodify_model_path',
                        type=str,
                        default="/yourpath_modify/BioMistral-7B_lasermodify",
                        help="Place where model weights are stored")  #

    parser.add_argument('--lasermodify_model_path_new',
                        type=str,
                        default="/yourpath_modify/BioMistral-7B_lasermodify_new",
                        help="Place where model weights are stored")  #

    parser.add_argument('--lasermodify_model_path_new1',
                        type=str,
                        default="/yourpath_modify/BioMistral-7B_lasermodify_new1",
                        help="Place where model weights are stored")  #

    parser.add_argument('--lasermodify_model_path_new2',
                        type=str,
                        default="/yourpath_modify/BioMistral-7B_lasermodify_new2",
                        help="Place where model weights are stored")  #

    parser.add_argument('--lasermodify_model_path_noreduce',
                        type=str,
                        default="/yourpath_modify/BioMistral-7B_lasermodify_noreduce",
                        help="Place where model weights are stored")  # LT

    parser.add_argument('--lasermodify_model_path_reduce',
                        type=str,
                        default="/yourpath_modify/BioMistral-7B_lasermodify_reduce",
                        help="Place where model weights are stored")  # LT


    parser.add_argument('--home_dir', type=str,
                        default="/yourpath/laser/BioMistral_results",
                        help='Directory where the data is')

    parser.add_argument('--csv_data_dir', type=str, default="/yourpath/BioMistralCSV",
                        help='Directory containing the MMLU college_medicine CSV dataset')  

    parser.add_argument('--split', type=str, default="test", choices=["dev", "test", "val"],
                        help='Which split of the dataset to use')

    parser.add_argument('--acc_threshold', type=float,default=0.680, help='acc_thresholdhold for accuracy')



    def str2intlist(s):
        return [int(item.strip()) for item in s.split(',')]
    def str2list(s):
        result = []
        for item in s.split(','):
            stripped_item = item.strip()
            if stripped_item.lower() == 'none':
                result.append(None)
            else:
                try:

                    result.append(float(stripped_item))
                except ValueError:
                    raise argparse.ArgumentTypeError(f"Invalid element: {stripped_item}. Must be a float or 'None'.")
        return result



    parser.add_argument('--layer_indices', type=str2intlist,
                        default=[i for i in range(32)],
                        help='A comma-separated list of layer indices, e.g., "0,1,2,3"')

    parser.add_argument('--rate_for_layer_indices', type=str2list,
                        default=[None for _ in range(32)],
                        help='A comma-separated list of rate, e.g., "5.0,7.5,None,None"')


    parser.add_argument("--save_dir", "-s", type=str, default="results")
    parser.add_argument("--ntrain", "-k", type=int, default=5)
    parser.add_argument("--ngpu", "-g", type=int, default=8)


    args = parser.parse_args()





    # Step 2: Load model and tokenizer
    llm_name = "BioMistral-7B"
    llm_path = args.model_path


    lasermodify_llm_path = args.lasermodify_model_path  #

    lasermodify_llm_path_new = args.lasermodify_model_path_new
    lasermodify_llm_path_new2 = args.lasermodify_model_path_new2

    layer_indices = args.layer_indices
    rate_for_layer_indices = args.rate_for_layer_indices

    lasermodify_llm_path_noreduce=args.lasermodify_model_path_noreduce
    lasermodify_llm_path_reduce=args.lasermodify_model_path_reduce

    acc_threshold=args.acc_threshold
    acc_threshold=float(acc_threshold)

    createFile(lasermodify_llm_path_noreduce)
    createFile(lasermodify_llm_path_reduce)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path,use_fast=False,add_bos_token=False, model_max_length=4096,padding_side="right",trust_remote_code=True)

    model_edit = AutoModelForCausalLM.from_pretrained(args.model_path,  torch_dtype=torch.float16, device_map="auto",trust_remote_code=True)


    config = AutoConfig.from_pretrained(llm_path)

    Original_model_parameters=count_parameters(model_edit)


    def print_weights(name, weight):
        print(f"{name} : {weight}")
        np.savetxt(f'/yourpath_modify/{name}.txt', weight.cpu().detach().numpy().reshape(-1, weight.shape[-1]), delimiter=',')

    home_dir = args.home_dir
    data_dir = args.csv_data_dir  # 使用新的 CSV 数据集路径
    save_dir = f"{home_dir}/{llm_name}/{args.intervention}/{args.lname}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    logger = Logger(save_dir=save_dir, fname=f"{llm_name}-log-{args.lnum}-{args.lname}-{args.rate}.txt")

    # Step 4: Create an experiment
    experiment = MistralExperiment(save_dir=save_dir, logger=logger)
    logger.log("=" * 50)
    logger.log(f"Created a new Experiment. Model {llm_name}")
    logger.log("=" * 50)
    for k, v in args.__dict__.items():
        logger.log(f">>>> Command line argument {k} => {v}")
    logger.log("=" * 50)


    model_edit, U, S, Vt=experiment.intervene(model=model_edit,
                                             tokenizer=tokenizer,
                                             args=args,
                                             llm_name=llm_name,
                                             layer_indices=layer_indices,
                                             rate_for_layer_indices=rate_for_layer_indices
                                             )


    logger.log("Experimented Completed.")



    print("Modified model parameters:", count_parameters(model_edit))

    subjects = sorted(
        [
            f.split("_test.csv")[0]
            for f in os.listdir(os.path.join(args.csv_data_dir, "test"))
            if "_test.csv" in f
        ]
    )

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(os.path.join(args.save_dir, "results_{}".format(args.model_path))):
        os.makedirs(os.path.join(args.save_dir, "results_{}".format(args.model_path)))

    all_cors = []
    subcat_cors = {
        subcat: [] for subcat_lists in subcategories.values() for subcat in subcat_lists
    }

    print('subcat_cors:',subcat_cors)
    cat_cors = {cat: [] for cat in categories}  #
    print('cat_cors:',cat_cors)
    acc_all=[]
    for subject in subjects:

        if 1:
            dev_df = pd.read_csv(
                os.path.join(args.csv_data_dir, "xxx", subject + "_xxx.csv"), header=None
            )[: args.ntrain]
            test_df = pd.read_csv(
                os.path.join(args.csv_data_dir, "yyy", subject + "_yyy.csv"), header=None
            )

            cors, acc, probs = eval(args, subject, model_edit, tokenizer, dev_df, test_df)  #这里的edit是原始模型，不是编辑过的

            acc_all.append(acc)
    acc_mean=np.mean(np.array(acc_all))
    print("Average accuracy {:.3f}".format(np.mean(np.array(acc_all))))



    processed_data = [str(item).replace('.', '') if item is not None else 'None' for item in rate_for_layer_indices]
    rate_for_layer_indices_str= '_'.join(processed_data)
    modelsubpath=lasermodify_llm_path_noreduce+'/%s'%rate_for_layer_indices_str
    subperform_output_file0 =lasermodify_llm_path_noreduce+'/performance_index.csv'
    subperform_output_file =modelsubpath+'/performance_index.csv'
    AccAllFile =lasermodify_llm_path_noreduce+'/performance_index_allandmean.csv'
    AccAllFile3f =lasermodify_llm_path_noreduce+'/performance_index_allandmean3f.csv'


    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')





    data = {
        'acc': [acc_mean],
        'rate_for_layer_indices_str': [rate_for_layer_indices_str],
        'createtime': [current_time]
    }
    df = pd.DataFrame(data)
    df.to_csv(subperform_output_file0, index=False)

    if acc_mean>=acc_threshold:

        model_edit.save_pretrained(modelsubpath) #
        tokenizer.save_pretrained(modelsubpath)
        output_file =lasermodify_llm_path_noreduce+'/performance_index_all.csv'


        df.to_csv(subperform_output_file, index=False)
        print(f"性能指标已保存到 {subperform_output_file}")
        print(df)


        ordered_subjects = ['clinical_knowledge', 'medical_genetics', 'anatomy', 'professional_medicine',
                            'college_biology', 'college_medicine', 'MedQA', 'MedQA-5_options',
                            'PubMedQA', 'MedMCQA', 'AllBioMedicine']

        sorted_acc_all = [acc_all[subjects.index(sub)] for sub in ordered_subjects]

        dfall = pd.DataFrame([sorted_acc_all], columns=ordered_subjects)
        dfall.to_csv(AccAllFile, index=False)

        print(dfall.to_string(index=False, formatters={col: '{:.3f}'.format for col in dfall.columns}))
        dfall.to_csv(AccAllFile3f, index=False, float_format='%.3f')





