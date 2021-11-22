# Written by Seungil Lee, Nov 22, 2021
# docker attach kairi_nvidia
# conda activate train

import random
import numpy as np
import argparse
import logging
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
import logging
from datasets import ContextCaptionDataset
from utils import accuracy, AverageMeter
from nltk.translate.bleu_score import corpus_bleu

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Data parameters
PATH = '/mnt/nas2/seungil/'  # folder with data files saved by create_input_files.py
data_name = 'FA'  # base name shared by data files

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead

# Training parameters
workers = 1  # for data-loading; right now, only 1 works with h5py
print_freq = 100  # print training/validation stats every __ batches
checkpoint = "BEST_checkpoint_FA.pth.tar" #1.66GB 
from_checkpoint_encoder = False 
from_checkpoint_decoder = False

##########################
""" vilbert module """
from types import SimpleNamespace
from easydict import EasyDict as edict
import yaml

from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.modeling_bert import BertModel
# it is from demo.py 
# its purpose is tokenizing a custom input data 
# here's no need 
# encoder = model
##########################

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--message", type=str, required=True, help="provide some detailed description of this test please"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="batch_size"
    )

    args_ = parser.parse_args()
    
    """
    From ViLBERT
    
    """
    
    args = SimpleNamespace(from_pretrained= PATH + "pretrained_model.bin",
                       bert_model="bert-base-uncased",
                       config_file="config/bert_base_6layer_6conect.json",
                       max_seq_length=101,
                       train_batch_size=1,
                       do_lower_case=True,
                       predict_feature=False,
                       seed=42,
                       num_workers=1,
                       baseline=False,
                       img_weight=1,
                       distributed=False,
                       objective=1,
                       visual_target=0,
                       dynamic_attention=False,
                       task_specific_tokens=True,
                       tasks='19',
                       save_name='',
                       in_memory=False,
                       batch_size=1,
                       local_rank=-1,
                       split='mteval',
                       clean_train_sets=True
                      )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    global checkpoint, data_name
            
    logger.info(f"Loaded from checkpoint: {checkpoint}")
    checkpoint = torch.load(PATH + 'checkpoints/' + checkpoint, map_location=str(device))
    decoder = checkpoint['decoder']
    encoder = checkpoint['encoder']

    n_gpu = torch.cuda.device_count()
    if n_gpu>0:
        torch.cuda.manual_seed_all(args.seed)

    logger.info(
        "device: {} n_gpu: {}".format(
            device, n_gpu
        )
    )

    if n_gpu>1:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)

    # Move to GPU, if available
    decoder = decoder.to(device)
    encoder = encoder.to(device)

    decoder.eval()
    encoder.eval()

    logger.info("*****Test Started*****")
    logger.info("")

    # Custom dataloaders
    test_loader = torch.utils.data.DataLoader(
        ContextCaptionDataset(
            "TASK19",
            dataroot=PATH,
            annotations_jsonpath=PATH+'jsonlines/FA.jsonline',
            split='val',
            features_h5path1 = PATH+'lmdbs/FA',
            features_h5path2 = '', #gt_image_features_reader='',
            tokenizer=tokenizer,
            bert_model=args.bert_model,
            ),
        batch_size=args_.batch_size, shuffle=True, num_workers=workers, pin_memory=True)


    # One epoch's validation
    test(test_loader=test_loader,
            encoder=encoder,
            decoder=decoder,
            tokenizer=tokenizer)

                                                                
def test(test_loader, encoder, decoder, tokenizer):
    """
    Performs one epoch's validation.

    :param test_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()

    start = time.time()
    
    references = list()  # references (true captions) for calculating BLEU-4 score
    hypothesis = list()  # hypothesis (predictions)

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, batch in enumerate(test_loader):
            # print(batch)
            features, spatials, image_mask, context, caption, input_mask, segment_ids, co_attention_mask, image_id, caplens = (
                        batch
        ) 

            task_tokens = context.new().resize_(context.size(0), 1).fill_(19) 

            
            # Move to GPU, if available
            context = context.to(device)
            features = features.to(device)
            spatials = spatials.to(device)
            segment_ids = segment_ids.to(device)
            input_mask = input_mask.to(device)
            image_mask = image_mask.to(device)
            co_attention_mask = co_attention_mask.to(device)
            task_tokens = task_tokens.to(device)

            # Forward prop.
            _, _, _, _, _, _, _, _, _, _, pooled_output = encoder(
                context,
                features,
                spatials,
                segment_ids,
                input_mask,
                image_mask,
                co_attention_mask,
                task_tokens,
            )
            
            pooled_output = pooled_output.to(device)
            caption = caption.to(device)
            # caplans = ~
            scores, caps_sorted, decode_lengths, alphas = decoder(pooled_output, caption, caplens)
            
            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]
            # print(f"targets shape from cap_sorted[:, 1:] : {targets.shape}") #torch.Size([32, 29])

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_packed, _, _, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True) 
                            #[32, 29, 30522] , torch.Size([32]) <= [tensor([29]), .. , tensor([29])]  
            
            targets_packed, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)


            # Keep track of metrics
            top5 = accuracy(scores_packed, targets_packed, 5)
            top5accs.update(top5, sum(decode_lengths)) 
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss_val:.4f} ({loss_avg:.4f})\t'
                      'Top-5 Accuracy {top5_val:.3f} ({top5_avg:.3f})\t'.format(i, len(test_loader), batch_time=batch_time,
                                                                                loss_val = float(losses.val), loss_avg = float(losses.avg), top5_val = float(top5accs.val), top5_avg=float(top5accs.avg)))    
            
             # References
            for j in range(targets.shape[0]): #16  #[batch size, max seq len?] == [16, 29]
                img_caps = targets[j].tolist() # validation dataset only has 1 unique caption per img
                img_caps = tokenizer.convert_ids_to_tokens(img_caps) # th) it has to be a one sentence
                # print(f"img_caps check! is it a sentence ? \n{img_caps}")
                clean_cap = [w for w in img_caps if w not in ["[PAD]","[CLS]","[SEP]"]]  # remove pad, start, and end # clean function
                img_captions = list(map(lambda c: clean_cap,img_caps)) # th) img_captions has to be a one clean sentence. 
                references.append([img_captions[0]])
                
            
            # hypothesis
            # preds.shape torch.Size([32, 29])
            _, preds = torch.max(scores, dim=2) ################# dim=1(changed) <- dim=2(original)
                # _, preds ==> values, and index respectively 
                # "dim =1" means it extracts 928 elements from 928 * 30522. that is, criterion is dim 1 which will be shrinked
            # print(f"predicted logits : {preds}")
            preds = preds.tolist() 
            preds_token = []
            for l in preds : 
                preds_token.append(tokenizer.convert_ids_to_tokens(l))
            temp_preds = list()
            
            for j, p in enumerate(preds_token):
                # print(f"iter : {j}, p in preds : {p}, p shape : {len(p)}")
                # print(f"decode_lengths : {decode_lengths}, len : {len(decode_lengths)}")
                # pred = p[:decode_lengths[j]] # decode_lenths is from decoder's 3rd output, like ... 29? 30? 
                pred = p[:decode_lengths[j]]
                pred = [w for w in pred if w not in ["[PAD]", "[CLS]","[SEP]"]]
                temp_preds.append(pred)  # remove pads, start, and end
            preds = temp_preds
            hypothesis.extend(preds)

            assert len(references) == len(hypothesis)
        
        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypothesis)
        
        print(
            '\n * LOSS - {loss_avg:.3f}, TOP-5 ACCURACY - {top5_avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss_avg=float(losses.avg),
                top5_avg=float(top5accs.avg),
                bleu=bleu4))
        print(f"references : {references}")
        print(f"hypothesis : {hypothesis}")

    return bleu4

if __name__ == '__main__':
    main()
