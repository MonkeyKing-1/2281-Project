import torch
import argparse
import contexttimer
from colorama import Fore, Style
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import logging
import sys
import random

from sampling import autoregressive_sampling, speculative_sampling, speculative_sampling_v2
from sampling.speculative_sampling import speculative_sampling, speculative_sampling_v2, speculative_sampling_v3, speculative_sampling_v4
from globals import Decoder
#from accelerate import Accelerator

from models.learner import LearnerModel
from models.drafting import ModelWrapper
from models.training import train_learner_with_target, get_distributions, sample_training_data
from torch.utils.data import Dataset, DataLoader
from datasetutils import EnhancedFeatureDataset, collate_fn
from datetime import datetime
from datasets import load_dataset
import csv
#import torch.multiprocessing as mp

wandb_installed = True
try:
    import wandb
except ImportError:
    wandb_installed = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# my local models
MODELZOO = {
    # llama-1
    # https://huggingface.co/PY007/TinyLlama-1.1B-step-50K-105b
    "llama1b": "/share_nfs/fangjiarui/root/code/hf_models/TinyLlama-1.1B-step-50K-105b",
    "llama7b": "/share_nfs/tianzhi/code/llama-7b",
    "llama30b": "/share_nfs/fangjiarui/root/code/hf_models/llama-30b-hf",
    "llama2-7b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-7b-hf",
    "llama2-70b" : "/share_nfs/fangjiarui/root/code/hf_models/llama-2-70b-hf",
    "bloom-560m": "/share_nfs/fangjiarui/root/code/hf_models/bloom-560m",
    "bloom-7b": "/share_nfs/fangjiarui/root/code/hf_models/bloomz-7b1",
    "baichuan-7b": "/share_nfs/duanqiyuan/models/source_models/hf/baichuan-7B",
    "baichuan-13b": "/share_nfs/duanqiyuan/models/source_models/hf/Baichuan-13B-Base",
}

drafter_models = {
    "pythia": ["EleutherAI/pythia-70m", "EleutherAI/pythia-160m", "EleutherAI/pythia-410m"],
    "bloomz": ["bigscience/bloomz-560m", "bigscience/bloom-560m","bigscience/bloomz-1b1", "bigscience/bloom-1b1"]
}

target_models = {
    "pythia": "EleutherAI/pythia-2.8b",
    "bloomz": "bigscience/bloomz-7b1"
}

def parse_file_name(s):
    i = 0
    while s[i] != '-':
        i+=1
    family = s[:i]
    i += 1
    l = []
    while True:
        l.append(int(s[i]))
        i += 1
        if s[i] == '-':
            break
        i += 1
    return family, l

def parse_arguments():
    parser = argparse.ArgumentParser(description='args for main.py')

    parser.add_argument('--input', type=str, default="Any recommendations for my holidays in Abu Dhabi?")
    parser.add_argument('--approx_model_name', type=str, default='bigscience/bloom-560m')
    parser.add_argument('--target_model_name', type=str, default='bigscience/bloomz-7b1')
    parser.add_argument('--student_model_name', type=str, default='bigscience/bloom-560m')
    parser.add_argument('--verbose', '-v', action='store_true', default=False, help='enable verbose mode')
    parser.add_argument('--seed', '-s', type=int, default=None, help='set a random seed, which can makes the result reproducible')
    parser.add_argument('--benchmark', '-b', action='store_true', default=False, help='show benchmark results.')
    parser.add_argument('--profiling', '-p', action='store_true', default=False, help='collect torch profiler results.')
    parser.add_argument('--max_tokens', '-M', type=int, default=20, help='max token number generated.')
    parser.add_argument('--gamma', '-g', type=int, default=4, help='guess time.')

    parser.add_argument('--mode', type=str, default='decode', choices=['decode', 'decode_v2', 'train_learner', 'create_dataset', 'distill'], 
                        help='Choose mode: decode, train_learner, create_dataset, or distill')
    parser.add_argument('--drafters', nargs='*', help='List of drafters', required=False)
    parser.add_argument('--sizes', nargs='*', help='List of size', required=False)
    parser.add_argument('--drafters_idx', nargs="*", help='List of drafter indices for training', required = False)
    parser.add_argument('--ptfile', type=str, help='ptfile', required=False)
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for learner training')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for learner training')
    parser.add_argument('--metric', type=str, default='kl', choices=['kl','l2', 'chi_squared', 'wasserstein', 'lk'], help='Distance metric for learner')
    parser.add_argument('--lk_k', type=int, default=1, help='Exponent k for lk distance (used if metric is lk)')

    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension size for the Learner')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of layers for the Learner')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate for the Learner')

    parser.add_argument('--wandb_project', type=str, default=None, help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name')
    parser.add_argument('--checkpoint_dir', type=str, default='learner-checkpoints', help='Directory to save the learner checkpoints')

    parser.add_argument('--dataset_name', type=str, default='wikitext', help='Name of the dataset family on HuggingFace')
    parser.add_argument('--dataset_config', type=str, default=None, help='Config of the dataset for instance wikitext-2-raw-v1')
    parser.add_argument('--dataset_split', type=str, default='train', choices=['train', 'validation', 'test'], help='Split of the dataset')
    parser.add_argument('--max_examples', type=int, default=50000, help='Maximum number of examples to process from the dataset')

    parser.add_argument('--lr_distillation', type=float, default=1e-5, help='Learning rate for distillation')
    parser.add_argument('--temperature', type=float, default=1.0, help='Temperature for distillation')
    parser.add_argument('--distillation_directory', type=str, help='Directory for the outputs of distillation', required=False)
    parser.add_argument('--save_interval', type=int, default=50, help='Interval for intermediate loss logging')

    args = parser.parse_args()
    return args

def color_print(text):
    print(Fore.RED + text + Style.RESET_ALL)
    
def benchmark(fn, print_prefix, use_profiler=True, *args, **kwargs):
    TEST_TIME = 10
    profile_filename = f"./profile_logs/{print_prefix}"
    
    with contexttimer.Timer() as t:
        if use_profiler:
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=0, warmup=1, active=2, repeat=1, skip_first=0),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_filename),
                record_shapes=False,
                profile_memory=False,
                # with_stack=True
            ) as prof:
                for _ in range(TEST_TIME): 
                    output = fn(*args, **kwargs)
                    prof.step()
        else:
            for _ in range(TEST_TIME): 
                output = fn(*args, **kwargs)

    print(f"\n [benchmark] {print_prefix}, tokens/sec: {len(output[0]) / t.elapsed / TEST_TIME}, {t.elapsed / TEST_TIME} sec generates {len(output[0])} tokens")

def generate(input_text, approx_model_name, target_model_name, num_tokens=20, gamma = 4,
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    tokenizer = AutoTokenizer.from_pretrained(approx_model_name, trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)
    
    print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
                                                       torch_dtype=torch.float16,
                                                       #device_map="auto",
                                                       device_map="cuda",
                                                       #load_in_8bit=True,
                                                       #offload_folder="offload",
                                                       trust_remote_code=True)
    large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                       torch_dtype=torch.float16,
                                                       #device_map="auto",
                                                       device_map="cuda",
                                                       #load_in_8bit=True,
                                                       #offload_folder="offload",
                                                       trust_remote_code=True)
    print("finish loading models")
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    top_k = 20
    top_p = 0.9

    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"large (target) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_large", use_profiling,
                  input_ids, large_model, num_tokens, top_k = top_k, top_p=top_p)

    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"small (approx) model autoregressive_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_small", use_profiling,
                  input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    
    torch.manual_seed(123)
    output = speculative_sampling_v2(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"deepmind's speculative_sampling: {generated_text}")   

    torch.manual_seed(123)
    output = speculative_sampling(input_ids, small_model, large_model, num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"google's speculative_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(speculative_sampling, "SP", use_profiling,
                  input_ids, small_model, large_model, max_len = num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed)

def generate_v2(input_text, ptfile, num_tokens=20, gamma = 10,
             random_seed = None, verbose = False, use_benchmark = False, use_profiling = False):
    # NOTE() approx_model_name and target_model_name should use the same tokenizer!
    
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    family, drafter_idxs = parse_file_name(ptfile)
    tokenizer = AutoTokenizer.from_pretrained(target_models[family], trust_remote_code=True)
  
    Decoder().set_tokenizer(tokenizer)
    # print(f"begin loading models: \n {approx_model_name} \n {target_model_name}")
    # small_model = AutoModelForCausalLM.from_pretrained(approx_model_name, 
    #                                                    torch_dtype=torch.float16,
    #                                                    #device_map="auto",
    #                                                    device_map="cuda",
    #                                                    #load_in_8bit=True,
    #                                                    #offload_folder="offload",
    #                                                    trust_remote_code=True)
    # large_model = AutoModelForCausalLM.from_pretrained(target_model_name, 
                                                    #    torch_dtype=torch.float16,
                                                    #    #device_map="auto",
                                                    #    device_map="cuda",
                                                    #    #load_in_8bit=True,
                                                    #    #offload_folder="offload",
                                                    #    trust_remote_code=True)
    large_model = ModelWrapper(target_models[family])
    small_models = [ModelWrapper(drafter_models[family][i]) for i in drafter_idxs]
    print("finish loading models")
    
    input_ids = tokenizer.encode(input_text, return_tensors='pt').to(torch_device)

    # top_k = 0
    # top_p = 0.0

    torch.manual_seed(123)
    output = autoregressive_sampling(input_ids, large_model, num_tokens)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    color_print(f"large (target) model autoregressive_sampling: {generated_text}")
    if use_benchmark:
        benchmark(autoregressive_sampling, "AS_large", use_profiling,
                  input_ids, large_model, num_tokens)

    # torch.manual_seed(123)
    # output = autoregressive_sampling(input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # color_print(f"small (approx) model autoregressive_sampling: {generated_text}")
    
    # if use_benchmark:
    #     benchmark(autoregressive_sampling, "AS_small", use_profiling,
    #               input_ids, small_model, num_tokens, top_k = top_k, top_p=top_p)
    
    # initialize model    
    # set learner model parameters
    input_dim = 4097 
    hidden_dim = 32
    L = len(small_models) # 
    num_layers = 25
    dropout = 0.2
    learner = LearnerModel(input_dim, hidden_dim, L, num_layers, dropout)

    # load weights
    state_dict = torch.load(ptfile) # EDIT: weights.pt
    learner.load_state_dict(state_dict)
    learner = learner.to(torch_device)
    
    if random_seed != None:
        torch.manual_seed(random_seed)
    else:
        torch.manual_seed(123)
    avg_loops = 0
    for i in range(100):
        torch.manual_seed(random_seed + i)
        loops, output = speculative_sampling_v3(input_ids, small_models, large_model, learner, num_tokens)
    # output = autoregressive_sampling(input_ids, small_models[1], num_tokens)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        color_print(f"our speculative_sampling: {generated_text}")
        avg_loops += loops
    avg_loops /= 100  
    print(avg_loops)

    
    # torch.manual_seed(123)
    # output = speculative_sampling_v2(input_ids, small_model, large_model, num_tokens, top_k = top_k, top_p=top_p, random_seed = random_seed)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # color_print(f"deepmind's speculative_sampling: {generated_text}")   

    # torch.manual_seed(123)
    # output = speculative_sampling(input_ids, small_model, large_model, num_tokens, gamma = gamma, top_k = top_k, top_p=top_p, random_seed = random_seed, verbose = verbose)
    # generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    # color_print(f"google's speculative_sampling: {generated_text}")
    
    if use_benchmark:
        benchmark(speculative_sampling_v3, "SP", use_profiling,
                  input_ids, small_models, large_model, learner, max_len = num_tokens, gamma = gamma, random_seed = random_seed)

def extract_texts_from_dataset(dataset, dataset_name):
    """
    Extract texts from the given dataset using the dataset_name argument
    Note that wikitext, pile-10k, openwebtext-10k, lambada, redpajama-v2 all use "text"
    Note that ag_news uses "description"
    """
    if dataset_name == "ag_news":
        texts = [item["description"] for item in dataset if item["description"].strip() != ""]
    else:
        texts = [item["text"] for item in dataset if item["text"].strip() != ""]
    return texts

if __name__ == "__main__":
    args = parse_arguments()

    torch.manual_seed(123)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    wandb_initialized = False

    if wandb_installed and args.wandb_project:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
        wandb_initialized = True

    if args.mode == 'decode':
        generate(args.input, args.approx_model_name, args.target_model_name, num_tokens=args.max_tokens, gamma=args.gamma,
                random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark)
    
    if args.mode == 'decode_v2':
        assert args.ptfile != None
        generate_v2(args.input, ptfile = args.ptfile, num_tokens=args.max_tokens, gamma=args.gamma,
                random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark)

    if args.mode == 'decode_v2':
        assert args.ptfile != None
        generate_v2(args.input, ptfile = args.ptfile, num_tokens=args.max_tokens, gamma=args.gamma,
                random_seed = args.seed, verbose=args.verbose, use_benchmark = args.benchmark)

    elif args.mode == 'train_learner':
        if not args.ptfile:
            raise ValueError("Please provide a pre-generated dataset file using --ptfile")

        print(f"Loading dataset from {args.ptfile}...")
        training_data = torch.load(args.ptfile)

        drafter_indices = args.drafters_idx
        if drafter_indices != None:
            drafter_indices = [int(d) for d in drafter_indices]
        else:
            raise ValueError("Please specify drafter indices using --drafters_idx")

        model_name_parts = args.target_model_name.split('/')
        model_family = model_name_parts[1] if len(model_name_parts) > 1 else args.target_model_name
        model_family = model_family.lower().split('-')[0]

        if "bloom" in model_family:
            input_dim = 4097
        elif "pythia" in model_family:
            input_dim = 2561
        else:
            raise ValueError(f"Unknown model_family: {model_family}, cannot determine input_dim.")

        print(f"Initializing Learner for {model_family} with input_dim={input_dim}...")
        L = len(drafter_indices)
        #Initialize the learner; note that bloom is 4097 and pythia is 2561 for the input dimension
        learner = LearnerModel(input_dim=input_dim, hidden_dim=args.hidden_dim, L=L, num_layers=args.num_layers, dropout=args.dropout).to(device)

        print("Training Learner model...")
        sizes = args.sizes
        sizes = [float(s) for s in sizes]
        print(f"Sizes is {sizes}")

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        if args.metric == 'lk':
            metric_name = f"lk{args.lk_k}"
        else:
            metric_name = args.metric

        drafter_indices_str = ",".join(map(str, drafter_indices))

        epoch_losses = train_learner_with_target(learner, drafter_indices, None, None, ptfile=args.ptfile,
                                                metric=args.metric, epochs=args.epochs,lr=1e-5, sizes = sizes, L=L,
                                                save_interval=args.save_interval,
                                                model_family=model_family,
                                                drafter_indices_str=drafter_indices_str,
                                                metric_name=metric_name,
                                                timestamp=timestamp,
                                                checkpoint_dir=args.checkpoint_dir,
                                                wandb_initialized=wandb_initialized)

        filename = f"{args.checkpoint_dir}/{model_family}-{drafter_indices_str}-{metric_name}-{timestamp}-weights.pt"
        torch.save(learner.state_dict(), filename)
        print(f"Learner has finished training and the model was saved to {filename}")

    elif args.mode == 'create_dataset':
        # if args.dataset_config is not None:
        #     raw_dataset = load_dataset(args.dataset_name, args.dataset_config)
        # else:
        #     raw_dataset = load_dataset(args.dataset_name)
        dataset_specs = [
            {
                "name": "stas/openwebtext-10k",
                "config": None,
                "split": "train",
                "max_examples": args.max_examples
            },
            {
                "name": "cimec/lambada",
                "config": None,
                "split": "validation",
                "max_examples": args.max_examples
            },
            {
                "name": "NeelNanda/pile-10k",
                "config": None,
                "split": "train",
                "max_examples": args.max_examples
            },
            {
                "name": "vilm/RedPajama-v2-small",
                "config": None,
                "split": "train",
                "max_examples": args.max_examples
            }
        ]

        all_texts = []
        for spec in dataset_specs:
            if spec["config"] is not None:
                raw_dataset = load_dataset(spec["name"], spec["config"])
            else:
                raw_dataset = load_dataset(spec["name"])

            dataset_split = raw_dataset[spec["split"]]
            limit = min(spec["max_examples"], len(dataset_split))
            dataset_split = dataset_split.select(range(limit))

            texts = extract_texts_from_dataset(dataset_split, spec["name"])
            all_texts.extend(texts)

        random.shuffle(all_texts)

        # dataset_split = raw_dataset[args.dataset_split]
        # limit = min(args.max_examples, len(dataset_split))
        # dataset_split = dataset_split.select(range(limit))

        # texts = extract_texts_from_dataset(dataset_split, args.dataset_name)

        target_model = ModelWrapper(args.target_model_name)

        tokenizer = target_model.tokenizer
        dataset = EnhancedFeatureDataset(tokenizer, target_model, texts, seq_len=128)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

        drafters = args.drafters
        drafters = [ModelWrapper(m) for m in drafters]
        L = len(drafters)
        sizes = args.sizes
        sizes = [float(s) for s in sizes]

        sample_training_data(drafters, target_model, data_loader, metric=args.metric, epochs=args.epochs, output=args.ptfile, k=args.lk_k, sizes=sizes)
        print(f"Offline dataset saved to {args.ptfile}")

    elif args.mode == 'distill':
        from models.training import distill_drafter_with_teacher
        teacher_model = ModelWrapper(args.target_model_name)
        student_model = ModelWrapper(args.student_model_name)
        if not args.distillation_directory:
            raise ValueError("Please provide a file to save the distilled model information using --distillation_directory")

        if args.dataset_config is not None:
            raw_dataset = load_dataset(args.dataset_name, args.dataset_config)
        else:
            raw_dataset = load_dataset(args.dataset_name)

        dataset_split = raw_dataset[args.dataset_split]
        limit = min(args.max_examples, len(dataset_split))
        dataset_split = dataset_split.select(range(limit))

        texts = extract_texts_from_dataset(dataset_split, args.dataset_name)

        tokenizer = teacher_model.tokenizer
        dataset = EnhancedFeatureDataset(tokenizer, teacher_model, texts, seq_len=128)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

        distill_epochs = args.epochs
        distill_lr = args.lr_distillation

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        teacher_base = args.target_model_name.replace('/', '_')
        student_base = args.student_model_name.replace('/', '_')
        dataset_desc = args.dataset_name

        distill_dir_name = f"{args.distillation_directory}/{teacher_base}_to_{student_base}_{dataset_desc}_temperature_{args.temperature}_{timestamp}"

        distill_drafter_with_teacher(student_model, teacher_model, data_loader, epochs=distill_epochs,
                                    temperature=args.temperature, lr=distill_lr, distillation_directory=distill_dir_name,
                                    wandb_project=args.wandb_project, wandb_run_name=args.wandb_run_name,
                                    wandb_initialized=wandb_initialized, save_interval=args.save_interval)
        print(f"Distilled student model saved to {distill_dir_name}")
