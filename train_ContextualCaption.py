# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
# Written by Beomjin Seo, 21/01/2022
# This code is based on ViLBERT train script and some github repos 

# Usage e.g.
>>> python train_ContextualCaption.py --file_path [DATA PATH] --decoder_with_bert_emb [True] --gpu_num [GPUNUM which will be used]
>>> python train_ContextualCaption.py --file_path /mnt/nas2/seungil/ --decoder_with_bert_emb True --gpu_num 7
"""

import argparse
import json
import logging
import os
import random
from io import open
import math
import sys

from time import gmtime, strftime
from timeit import default_timer as timer

import numpy as np
from tqdm import tqdm, trange

import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from pytorch_transformers.modeling_bert import BertModel
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

import vilbert.utils as utils
from vilbert.datasets import ConceptCapLoaderTrain, ConceptCapLoaderVal
from vilbert.vilbert import BertForMultiModalPreTraining, BertConfig
import torch.distributed as dist

from vilbert.datasets import ContextCaptionDataset
from models import DecoderWithAttention, DecoderWithBertEmbedding
from nltk.translate.bleu_score import corpus_bleu
from torch.nn.utils.rnn import pack_padded_sequence


PAD = 0
CLS = 101
SEP = 102
UNK = 100

# os.environ["CUDA_VISIBLE_DEVICES"] = '4, 5, 6, 7'


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--file_path",
        default='/mnt/nas2/seungil/',
        type=str,
        help="The input train corpus.",
    )
    parser.add_argument(
        "--from_pretrained",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-base-uncased, roberta-base, roberta-large, ",
    )
    parser.add_argument(
        "--bert_model",
        default="bert-base-uncased",
        type=str,
        help="Bert pre-trained model selected in the list: bert-base-uncased, "
        "bert-large-uncased, roberta-base",
    )
    parser.add_argument(
        "--output_dir",
        default="save_ContextualCaption",
        type=str,
        # required=True,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/bert_base_6layer_6conect.json",
        help="The config file which specified the model details.",
    )
    ## Other parameters
    parser.add_argument(
        "--max_seq_length",
        default=150,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--train_batch_size",
        default=8,
        type=int,
        help="Total batch size for training.",
    )
    parser.add_argument(
        "--learning_rate",
        default=1e-4,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--num_train_epochs",
        default=20,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--start_epoch",
        default=0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--img_weight", default=1, type=float, help="weight for image loss"
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
    )
    parser.add_argument(
        "--on_memory",
        action="store_true",
        help="Whether to load train samples into memory or use disk",
    )
    parser.add_argument(
        "--do_lower_case",
        type=bool,
        default=True,
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="local_rank for distributed training on gpus",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumualte before performing a backward/update pass.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=0,
        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
        "0 (default value): dynamic loss scaling.\n"
        "Positive power of 2: static loss scaling value.\n",
    )
    parser.add_argument(
        "--dynamic_attention",
        action="store_true",
        help="whether use dynamic attention.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=16,
        help="Number of workers in the dataloader.",
    )
    parser.add_argument(
        "--save_name", default="", type=str, help="save name for training."
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Wheter to use the baseline model (single bert).",
    )
    parser.add_argument(
        "--freeze",
        default=-1,
        type=int,
        help="till which layer of textual stream of vilbert need to fixed.",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="whether use chunck for parallel training.",
    )
    parser.add_argument(
        "--without_coattention", action="store_true", help="whether pair loss."
    )
    parser.add_argument(
        "--visual_target",
        default=0,
        type=int,
        help="which target to use for visual branch. \
        0: soft label, \
        1: regress the feature, \
        2: NCE loss.",
    )

    # parser.add_argument(
    #     "--objective",
    #     default=0,
    #     type=int,
    #     help="which objective to use \
    #     0: with ICA loss, \
    #     1: with ICA loss, for the not aligned pair, no masking objective, \
    #     2: without ICA loss, do not sample negative pair.",
    # )
    # parser.add_argument(
    #     "--num_negative", default=255, type=int, help="num of negative to use"
    # )

    parser.add_argument(
        "--resume_file", default="", type=str, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )

    parser.add_argument(
        "--best_bleu4", default=0., type=float, help="initialize bleu4 value"
    )

    parser.add_argument(
        "--dec_emb_dim", default=512, type=int, help="decoder embedding dim"
    )
    parser.add_argument(
        "--dec_attention_dim", default=512, type=int, help="decoder attention dim"
    )
    parser.add_argument(
        "--dec_decoder_dim", default=512, type=int, help="decoder dim"
    )
    parser.add_argument(
        "--dec_dropout", default=0.3, type=float, help="decoder dropout rate"
    )
    parser.add_argument(
        "--gpu_num", default=7, type=int, help="GPU number which will be used"
    )
    parser.add_argument(
        "--decoder_with_bert_emb", default=False, type=bool
    )

    args = parser.parse_args()

    # if args.baseline:
    #     from pytorch_pretrained_bert.modeling import BertConfig
    #     from vilbert.basebert import BertForMultiModalPreTraining
    # else:
    #     from vilbert.vilbert import BertForMultiModalPreTraining, BertConfig

    if args.baseline:
        from pytorch_transformers.modeling_bert import BertConfig
        from vilbert.basebert import BaseBertForVLTasks
    else:
        from vilbert.vilbert import BertConfig
        from vilbert.vilbert import VILBertForVLTasks

    if args.save_name:
        prefix = "-" + args.save_name
    else:
        prefix = ""

    timeStamp = args.config_file.split("/")[1].split(".")[0] + prefix
    savePath = os.path.join(args.output_dir, timeStamp)

    bert_weight_name = json.load(
        open("config/" + args.from_pretrained + "_weight_name.json", "r")
    )

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            f"cuda:{args.gpu_num}" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        torch.cuda.set_device(f"cuda:{args.gpu_num}")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device(f"cuda:{args.gpu_num}", args.local_rank)
        torch.cuda.set_device(f"cuda:{args.gpu_num}")
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")

    logger.info(
        "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            device, n_gpu, bool(args.local_rank != -1), args.fp16
        )
    )

    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True

    if default_gpu:
        if not os.path.exists(savePath):
            os.makedirs(savePath)

    config = BertConfig.from_json_file(args.config_file)

    if default_gpu:
        # save all the hidden parameters.
        with open(os.path.join(savePath, "command.txt"), "w") as f:
            print(args, file=f)  # Python 3.x
            print("\n", file=f)
            print(config, file=f)

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    cache = 5000
    if dist.is_available() and args.local_rank != -1:
        num_replicas = dist.get_world_size()
        args.train_batch_size = args.train_batch_size // num_replicas
        args.num_workers = args.num_workers // num_replicas
        cache = cache // num_replicas

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )
    num_train_optimization_steps = None

    train_dataset = DataLoader(
        ContextCaptionDataset(
            "TASK19",
            dataroot=args.file_path,
            annotations_jsonpath= args.file_path + 'jsonlines/' + "train" + '.jsonlines',
            split='train',
            features_h5path1 = args.file_path + 'lmdbs/' + "train" + ".lmdb",
            features_h5path2 = '', 
            tokenizer=tokenizer,
            bert_model=args.bert_model,
            ),
        batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    validation_dataset = DataLoader(
        ContextCaptionDataset(
            "TASK19",
            dataroot=args.file_path,
            annotations_jsonpath= args.file_path + 'jsonlines/' + "val" + '.jsonlines',
            split='val',
            features_h5path1 = args.file_path + 'lmdbs/' + "val" + ".lmdb",
            features_h5path2 = '',
            tokenizer=tokenizer,
            bert_model=args.bert_model,
            ),
        batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    num_train_optimization_steps = int(
        len(train_dataset)
        / args.train_batch_size
        / args.gradient_accumulation_steps
    ) * (args.num_train_epochs - args.start_epoch)

    task_names = ["Contextual_Caption"]
    task_ids = ["TASK19"]
    task_num_iters = {"TASK19": len(train_dataset) / args.train_batch_size}

    logdir = os.path.join("logs", timeStamp)
    if default_gpu:
        tbLogger = utils.tbLogger(
            logdir,
            savePath,
            task_names,
            task_ids,
            task_num_iters,
            args.gradient_accumulation_steps,
        )

    if args.visual_target == 0:
        config.v_target_size = 1601
        config.visual_target = args.visual_target
    else:
        config.v_target_size = 2048
        config.visual_target = args.visual_target

    if "roberta" in args.bert_model:
        config.model = "roberta"

    if args.freeze > config.t_biattention_id[0]:
        config.fixed_t_layer = config.t_biattention_id[0]

    if args.without_coattention:
        config.with_coattention = False

    if args.dynamic_attention:
        config.dynamic_attention = True

    num_labels = 0
    if args.baseline : 
        print("BaseBert is loaded.")
        model = BaseBertForVLTasks.from_pretrained(
            args.from_pretrained,
            config=config,
            num_labels=num_labels,
            default_gpu=default_gpu,
        )
    else:
        print("VILBert is loaded.")
        model = VILBertForVLTasks.from_pretrained(
            args.from_pretrained,
            config=config,
            num_labels=num_labels,
            default_gpu=default_gpu,
        )

    if args.decoder_with_bert_emb : 
        BertForDecoder = BertModel.from_pretrained(args.bert_model).to(device)
        BertForDecoder.eval()
        decoder = DecoderWithBertEmbedding(vocab_size=30522, use_glove=False, use_bert=True, tokenizer=tokenizer, BertModel=BertForDecoder)
        
    else : 
        decoder = DecoderWithAttention(attention_dim=args.dec_attention_dim,
                                       embed_dim=args.dec_emb_dim,
                                       decoder_dim=args.dec_decoder_dim,
                                       vocab_size=30522,
                                       dropout=args.dec_dropout)


    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    if args.freeze != -1:
        bert_weight_name_filtered = []
        for name in bert_weight_name:
            if "embeddings" in name:
                bert_weight_name_filtered.append(name)
            elif "encoder" in name:
                layer_num = name.split(".")[2]
                if int(layer_num) <= args.freeze:
                    bert_weight_name_filtered.append(name)

        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if key[12:] in bert_weight_name_filtered:
                value.requires_grad = False

        if default_gpu:
            print("filtered weight")
            print(bert_weight_name_filtered)

    if not args.from_pretrained:
        param_optimizer = list(model.named_parameters())
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
    else:
        optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if value.requires_grad:
                if key[12:] in bert_weight_name:
                    lr = args.learning_rate * 0.1
                else:
                    lr = args.learning_rate

                if any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.0}
                    ]

                if not any(nd in key for nd in no_decay):
                    optimizer_grouped_parameters += [
                        {"params": [value], "lr": lr, "weight_decay": 0.01}
                    ]

        if default_gpu:
            print(
                len(list(model.named_parameters())), len(optimizer_grouped_parameters)
            ) 
            # 588 588

    # set different parameters for vision branch and lanugage branch.
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )

        optimizer = FusedAdam(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            bias_correction=False,
            max_grad_norm=1.0,
        )
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon,
            betas=(0.9, 0.98),
        )

    scheduler = WarmupLinearSchedule(
        optimizer,
        warmup_steps=args.warmup_proportion * num_train_optimization_steps,
        t_total=num_train_optimization_steps,
    )

    criterion = torch.nn.CrossEntropyLoss().to(device)
    

    startIterID = 0
    global_step = 0
    start_epoch = int(args.start_epoch)
    best_bleu4 = float(args.best_bleu4)

    if args.resume_file != "" and os.path.exists(args.resume_file):
        checkpoint = torch.load(args.resume_file, map_location="cpu")
        new_dict = {}
        for attr in checkpoint["model_state_dict"]:
            if attr.startswith("module."):
                new_dict[attr.replace("module.", "", 1)] = checkpoint[
                    "model_state_dict"
                ][attr]
            else:
                new_dict[attr] = checkpoint["model_state_dict"][attr]
        model.load_state_dict(new_dict)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint["global_step"]
        start_epoch = int(checkpoint["epoch_id"]) +1 
        best_bleu4 = float(checkpoint["best_bleu4"])
        decoder = checkpoint["decoder"]
        del checkpoint

    # model.cuda()
    # decoder.cuda()

    # model.to(device)
    # decoder.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                # state[k] = v.cuda()
                state[k] = v.to(device)

    if args.fp16:
        model.half()
        decoder.half()
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model)
    elif n_gpu > 1:
        try : 
            # model = torch.nn.DataParallel(model, output_device=4)
            # decoder = torch.nn.DataParallel(decoder, output_device=4)
            # model.cuda()
            # decoder.cuda()
            model.to(device)
            decoder.to(device)
        except : 
            print("only one gpu")
            


    if default_gpu:
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)

    print("Traning...")
    for epochId in range(start_epoch, int(args.num_train_epochs)):
        model.train()
        decoder.train()
        for step, batch in enumerate(train_dataset):

            iterId = startIterID + step + (epochId * len(train_dataset))
            # image_ids = batch[-1]
            # batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch[:-1])

            features, spatials, image_mask, context, caption, context_input_mask, context_segment_ids, co_attention_mask, image_id, caplens = (
                        batch
            ) 
            task_tokens = context.new().resize_(context.size(0), 1).fill_(19).to(device)

            # Move to GPU, if available
            
            features = features.to(device)
            spatials = spatials.to(device)
            image_mask = image_mask.to(device)
            context = context.to(device)
            caption = caption.to(device)
            context_input_mask = context_input_mask.to(device)
            context_segment_ids = context_segment_ids.to(device)
            co_attention_mask = co_attention_mask.to(device)
            # image_id = image_id.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            _, _, _, _, _, _, _, _, _, _, pooled_output = model(
                context, # input txt
                features, # input imgs
                spatials, # img loc
                context_segment_ids, # token type id 
                context_input_mask, # text attention mask
                image_mask, # img attention mask 
                co_attention_mask, # co attention mask 
                None, #task_tokens, # default = None
            )

            scores, caps_sorted, decode_lengths, alphas = decoder(pooled_output, caption, caplens)
            scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]

            targets = caps_sorted[:, 1:]
            targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

            loss = criterion(scores_packed, targets_packed).to(device)
            loss += ((1. - alphas.sum(dim=1)) ** 2).mean() # ??

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    lr_this_step = args.learning_rate * warmup_linear(
                        global_step / num_train_optimization_steps,
                        args.warmup_proportion,
                    )
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr_this_step

                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if (
                step % (20 * args.gradient_accumulation_steps) == 0
                and step != 0
                and default_gpu
            ):
                print(f"Iter : {iterId}\tEp : {epochId}\tLoss : {loss}")


                if step % 200 == 0 : 

                    references, hypothesis = [], []

                    # print(f" RRR check targets : {targets}")
                    # print(f"to token : {tokenizer.convert_ids_to_tokens(targets)}")
                    
                    # References
                    for j in range(targets.shape[0]):
                        img_caps = targets[j].tolist() # validation dataset only has 1 unique caption per img
                        # print(f"  ### check img_caps : {img_caps}")
                        ### check img_caps : [16925, 10239, 1999, 1996, 5934, 2537, 1997, 15333, 4371, 8671, 1006, 4537, 1007, 1010, 2019, 7291, 4482, 2377, 1012, 2009, 2001, 2101, 5967, 2005, 1037, 4260, 2143, 2021, 0]
                        img_caps = tokenizer.convert_ids_to_tokens(img_caps)
                        clean_cap = [w for w in img_caps if w not in ["[PAD]", "[CLS]", "[SEP]"]]  # remove pad, start, and end
                        # img_captions = list(map(lambda c: clean_cap, img_caps))
                        references.append([clean_cap])

                    # Hypotheses
                    _, preds = torch.max(scores, dim=2)
                    preds = preds.tolist()
                    preds_token = []
                    temp_preds = []

                    for l in preds : 
                        preds_token.append(tokenizer.convert_ids_to_tokens(l))

                    for j, p in enumerate(preds_token):
                        pred = p[:decode_lengths[j]]
                        pred = [w for w in pred if w not in ["[PAD]", "[CLS]", "[SEP]"]]
                        temp_preds.append(pred)  # remove pads, start, and end
                    preds = temp_preds
                    hypothesis.extend(preds)

                    assert len(references) == len(hypothesis)

                    print(f"references :\n{references}")
                    print(f"hypothesis :\n{hypothesis}")
                                                
                    bleu4 = corpus_bleu(references, hypothesis)
                    print(f"bleu4 : {bleu4}")
                

        # Do the evaluation
        torch.set_grad_enabled(False)
        numBatches = len(validation_dataset)

        model.eval()
        decoder.eval()

        references, hypothesis = [], []

        print("Validating...")
        for step, batch in enumerate(validation_dataset):
            # image_ids = batch[-1]
            # batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch[:-1])

            # batch = batch.to(device)

            features, spatials, image_mask, context, caption, context_input_mask, context_segment_ids, co_attention_mask, image_id, caplens = (
                        batch
            ) 

            task_tokens = context.new().resize_(context.size(0), 1).fill_(19).to(device)

            features = features.to(device)
            spatials = spatials.to(device)
            image_mask = image_mask.to(device)
            context = context.to(device)
            caption = caption.to(device)
            context_input_mask = context_input_mask.to(device)
            context_segment_ids = context_segment_ids.to(device)
            co_attention_mask = co_attention_mask.to(device)
            # image_id = image_id.to(device)
            caplens = caplens.to(device)

            # Forward prop.
            _, _, _, _, _, _, _, _, _, _, pooled_output = model(
                context, # input txt
                features, # input imgs
                spatials, # img loc
                context_segment_ids, # token type id : (separates segment A from segment B)
                context_input_mask, # text attention mask : (input_mask annotates real token sequence from padding)
                image_mask, # img attention mask 
                co_attention_mask, # co attention mask 
                None, #task_tokens, # default = None
            )

            scores, caps_sorted, decode_lengths, alphas = decoder(pooled_output, caption, caplens)
            scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]

            targets = caps_sorted[:, 1:]
            targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

            loss = criterion(scores_packed, targets_packed)
            loss += ((1. - alphas.sum(dim=1)) ** 2).mean()

            # References
            for j in range(targets.shape[0]):
                img_caps = targets[j].tolist() # validation dataset only has 1 unique caption per img
                # print(f"  ### check img_caps : {img_caps}")
                ### check img_caps : [16925, 10239, 1999, 1996, 5934, 2537, 1997, 15333, 4371, 8671, 1006, 4537, 1007, 1010, 2019, 7291, 4482, 2377, 1012, 2009, 2001, 2101, 5967, 2005, 1037, 4260, 2143, 2021, 0]
                img_caps = tokenizer.convert_ids_to_tokens(img_caps)
                clean_cap = [w for w in img_caps if w not in ["[PAD]", "[CLS]", "[SEP]"]]  # remove pad, start, and end
                # img_captions = list(map(lambda c: clean_cap, img_caps))
                references.append([clean_cap])

            # Hypotheses
            _, preds = torch.max(scores, dim=2)
            preds = preds.tolist()
            preds_token = []
            temp_preds = []

            for l in preds : 
                preds_token.append(tokenizer.convert_ids_to_tokens(l))

            for j, p in enumerate(preds_token):
                pred = p[:decode_lengths[j]]
                pred = [w for w in pred if w not in ["[PAD]", "[CLS]", "[SEP]"]]
                temp_preds.append(pred)  # remove pads, start, and end
            preds = temp_preds
            hypothesis.extend(preds) 

        assert len(references) == len(hypothesis)

        print(f"references :\n{references}")
        print(f"hypothesis :\n{hypothesis}")
                                    
        recent_bleu4 = corpus_bleu(references, hypothesis)          

        # Calculate BLEU-4 scores
        # bleu1 = corpus_bleu(references, hypothesis, weights=(1,0,0,0))
        # bleu2 = corpus_bleu(references, hypothesis, weights=(0,1,0,0))
        # bleu3 = corpus_bleu(references, hypothesis, weights=(0,0,1,0))                
        # recent_bleu4 = corpus_bleu(references, hypothesis, weights=(0,0,0,1))

        # print(f"bleu1 : {bleu1}")
        # print(f"bleu2 : {bleu2}")
        # print(f"bleu3 : {bleu3}")
        print(f"bleu4 : {recent_bleu4}")
        # print(f"references : {references}")
        # print(f"hypothesis : {hypothesis}")


        if default_gpu:
            # ave_score = tbLogger.showLossValCC()
            print(f"Iter : {iterId}\tEp : {epochId}\tVal Loss : {loss}")
            

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        torch.set_grad_enabled(True)

        if default_gpu and is_best:
            # Save a trained model
            logger.info("** ** * Saving fine - tuned model ** ** * ")
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Only save the model it-self
            output_model_file = os.path.join(
                savePath, "pytorch_model_" + str(epochId) + ".bin"
            )
            output_checkpoint = os.path.join(
                savePath, "BEST_pytorch_ckpt_" + str(epochId) + ".pth.tar"
            )
            torch.save(model_to_save.state_dict(), output_model_file)
            torch.save(
                {
                    "model_state_dict": model_to_save.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "global_step": global_step,
                    "epoch_id": epochId, 
                    "decoder": decoder, 
                    "bleu4" : recent_bleu4, 
                    "best_bleu4" : best_bleu4
                },
                output_checkpoint
            )   

    if default_gpu:
        tbLogger.txt_close()


if __name__ == "__main__":

    main()
