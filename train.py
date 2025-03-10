# docker attach kairi_nvidia
# conda activate train

import random
import numpy as np
import logging
import time
from requests.api import get
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from models import DecoderWithBertEmbedding 
import logging
from datasets import ContextCaptionDataset
from utils import save_checkpoint, adjust_learning_rate, accuracy, count_parameters, clip_gradient, AverageMeter
from nltk.translate.bleu_score import corpus_bleu
import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead


PATH = '/mnt/nas2/seungil/'  # folder with data files saved by create_input_files.py


##########################
""" vilbert module """
from types import SimpleNamespace

from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.modeling_bert import BertModel
##########################

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def get_args():
    args = SimpleNamespace(from_pretrained= PATH + "pretrained_model.bin",
                        bert_model="bert-base-uncased",
                        config_file="config/bert_base_6layer_6conect.json",
                        train='train',
                        val='val',
                        do_lower_case=True,
                        predict_feature=False,
                        seed=42,
                        workers=16,
                        baseline=False,
                        dynamic_attention=False,
                        task_specific_tokens=True,
                        batch_size=128,
                        save_name = 'FA,GA,AA0~AA7',  # base name shared by data files
                        start_epoch = 0,
                        epochs = 120,  # number of epochs to train for (if early stopping is not triggered)
                        epochs_since_improvement = 0,  # keeps track of number of epochs since there's been an improvement in validation BLEU
                        encoder_lr = 1e-4,  # learning rate for encoder if fine-tuning
                        decoder_lr = 4e-3,  # learning rate for decoder #4e-4
                        grad_clip = 5.,  # clip gradients at an absolute value of
                        alpha_c = 1.,  # regularization parameter for 'doubly stochastic attention', as in the paper
                        best_bleu4 = 0.,  # BLEU-4 score right now
                        print_freq = 100,  # print training/validation stats every __ batches
                        fine_tune_encoder = False,  # fine-tune encoder?
                        device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu"),  # sets device for model and PyTorch tensors
                        checkpoint = None #"BEST_checkpoint_FA_dataset.pth.tar"
    )
    return args

def main():    
    """
    From ViLBERT
    
    """
    args = get_args()

    if args.baseline:
        print("when baseline is True")
        from pytorch_transformers.modeling_bert import BertConfig
        from vilbert.basebert import BaseBertForVLTasks
    else:
        from vilbert.vilbert import BertConfig
        from vilbert.vilbert import VILBertForVLTasks
    
    config = BertConfig.from_json_file(args.config_file)
    
    timeStamp = args.from_pretrained.split('/')[-1] + '-' + args.save_name
    config = BertConfig.from_json_file(args.config_file)
    default_gpu=True
    
    if args.predict_feature:
        config.v_target_size = 2048
        config.predict_feature = True
    else:
        config.v_target_size = 1601
        config.predict_feature = False
    
    if args.task_specific_tokens:
        config.task_specific_tokens = True    
    
    if args.dynamic_attention:
        config.dynamic_attention = True
    
    config.visualization = True
    num_labels = 3129
    
    if args.baseline:
        encoder = BaseBertForVLTasks.from_pretrained(
            args.from_pretrained, config=config, num_labels=num_labels, default_gpu=default_gpu
            )
    else:
        encoder = VILBertForVLTasks.from_pretrained(
            args.from_pretrained, config=config, num_labels=num_labels, default_gpu=default_gpu
            )
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )
    
    
    """
    Training and validation.
    """

    global best_bleu4, epochs_since_improvement, checkpoint, start_epoch, fine_tune_encoder
    start_epoch = args.start_epoch
    epochs = args.epochs
    epochs_since_improvement = args.epochs_since_improvement
    best_bleu4 = args.best_bleu4
    checkpoint = args.checkpoint
            
    if checkpoint is None:
        # Load pre-trained model (weights)
        BertForDecoder = BertModel.from_pretrained(args.bert_model).to(args.device)
        BertForDecoder.eval()
        decoder = DecoderWithBertEmbedding(vocab_size=30522,use_glove=False, use_bert=True, tokenizer=tokenizer, BertModel=BertForDecoder)
        decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()),
                                             lr=args.decoder_lr)
        # encoder.fine_tune(fine_tune_encoder)
        encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                             lr=args.encoder_lr) if args.fine_tune_encoder else None
    else: 
        logger.info(f"Loaded from checkpoint: {checkpoint}")
        checkpoint = torch.load(PATH + 'checkpoints/' + checkpoint, map_location=str(args.device))
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_bleu4 = checkpoint['bleu-4']
        decoder = checkpoint['decoder']
        decoder_optimizer = checkpoint['decoder_optimizer']
        encoder = checkpoint['encoder']
        encoder_optimizer = checkpoint['encoder_optimizer']
        if args. fine_tune_encoder is True and encoder_optimizer is None:
            # encoder.fine_tune(fine_tune_encoder)
            encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                                 lr=args.encoder_lr)
    
    n_gpu = torch.cuda.device_count()
    # if n_gpu>0:
    #     torch.cuda.manual_seed_all(args.seed)

    logger.info(
        "device: {} n_gpu: {}".format(
            args.device, n_gpu
        )
    )

    if n_gpu>1:
        encoder = nn.DataParallel(encoder,device_ids = [4,5,6,7])
        decoder = nn.DataParallel(decoder,device_ids = [4,5,6,7])
    # Move to GPU, if available
    decoder = decoder.to(args.device)
    encoder = encoder.to(args.device)

    encoder.eval()

    criterion = nn.CrossEntropyLoss().to(args.device)

    # Custom dataloaders
    train_loader = torch.utils.data.DataLoader(
        ContextCaptionDataset(
            "TASK19",
            dataroot=PATH,
            annotations_jsonpath= PATH + 'jsonlines/' + args.train + '.jsonline',
            split='train',
            features_h5path1 = PATH + 'lmdbs/' + args.train,
            features_h5path2 = '', 
            tokenizer=tokenizer,
            bert_model=args.bert_model,
            ),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        ContextCaptionDataset(
            "TASK19",
            dataroot=PATH,
            annotations_jsonpath= PATH + 'jsonlines/' + args.val + '.jsonline',
            split='val',
            features_h5path1 = PATH + 'lmdbs/' + args.val,
            features_h5path2 = '',
            tokenizer=tokenizer,
            bert_model=args.bert_model,
            ),
        batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    logger.info("***** Running training *****")
    logger.info("  Batch size = %d", args.batch_size)
    logger.info("  Num steps = %d", epochs)

    logger.info("*****Total Number of Parameters*****")
    logger.info("  Encoder # = %d", count_parameters(encoder))
    logger.info("  Decoder # = %d", count_parameters(decoder))

    # Epochs
    for epoch in range(start_epoch, epochs):
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(decoder_optimizer, 0.8)
            if args.fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              decoder=decoder,
              criterion=criterion,
              encoder_optimizer=encoder_optimizer,
              decoder_optimizer=decoder_optimizer,
              epoch=epoch)

        # One epoch's validation
        recent_bleu4 = validate(val_loader=val_loader,
                                encoder=encoder,
                                decoder=decoder,
                                criterion=criterion,
                                tokenizer=tokenizer)

        # Check if there was an improvement
        is_best = recent_bleu4 > best_bleu4
        best_bleu4 = max(recent_bleu4, best_bleu4)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(args.save_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer,
                        decoder_optimizer, recent_bleu4, is_best)


def train(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch):
    """
    Performs one epoch's training.

    :param train_loader: DataLoader for training data
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :param encoder_optimizer: optimizer to update encoder's weights (if fine-tuning)
    :param decoder_optimizer: optimizer to update decoder's weights
    :param epoch: epoch number
    """
    args = get_args()

    decoder.train()  # train mode (dropout and batchnorm is used)
    # encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()

    # Batches
    for i, batch in enumerate(train_loader):
        # features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = (
        #                 batch
        # ) 
        
        # I think this version is more appropriate, right? 
        features, spatials, image_mask, context, caption, input_mask, segment_ids, co_attention_mask, image_id, caplens = (
                        batch
        ) 

        task_tokens = context.new().resize_(context.size(0), 1).fill_(19)

        data_time.update(time.time() - start)

        # Move to GPU, if available
        context = context.to(args.device)
        features = features.to(args.device)
        spatials = spatials.to(args.device)
        segment_ids = segment_ids.to(args.device)
        input_mask = input_mask.to(args.device)
        image_mask = image_mask.to(args.device)
        co_attention_mask = co_attention_mask.to(args.device)
        task_tokens = task_tokens.to(args.device)
        # Forward prop.
        _, _, _, _, _, _, _, _, _, _, pooled_output = encoder(
            context, # input txt
            features, # input imgs
            spatials, # img loc
            segment_ids, # token type id 
            input_mask, # text attention mask
            image_mask, # img attention mask 
            co_attention_mask, # co attention mask 
            task_tokens, # default = None
        )

        pooled_output = pooled_output.to(args.device)
        caption = caption.to(args.device)
        # caplens = (torch.tensor([32,len(caption)])).to(device)
        # scores, caps_sorted, decode_lengths, alphas, sort_ind = decoder(pooled_output, caption, caplens)
        scores, caps_sorted, decode_lengths, alphas = decoder(pooled_output, caption, caplens)

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]

        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores, _, _, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _, _, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)

        # Calculate loss
        loss = criterion(scores, targets).to(args.device) # .to(device)

        # Add doubly stochastic attention regularization
        loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

        # Back prop.
        decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if args.grad_clip is not None:
            clip_gradient(decoder_optimizer, args.grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, args.grad_clip)

        # Update weights
        decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = accuracy(scores, targets, 5)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
        batch_time.update(time.time() - start)

        start = time.time()
        
        # Print status
        if i % args.print_freq == 0:
            print(f"Epoch: [{epoch}][{i}/{len(train_loader)}]")
            print(f"Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})")
            print(f"Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})")
            print(f"Loss {float(losses.val):.4f} ({float(losses.avg):.4f})")
            print(f"Top-5 Accuracy {float(top5accs.val):.3f} ({float(top5accs.avg):.3f})")       
                                                                
def validate(val_loader, encoder, decoder, criterion, tokenizer):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """

    args = get_args()

    decoder.eval()  # eval mode (no dropout or batchnorm)
    if encoder is not None:
        encoder.eval()

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
        for i, batch in enumerate(val_loader):
            # print(batch)
            features, spatials, image_mask, context, caption, input_mask, segment_ids, co_attention_mask, image_id, caplens = (
                        batch
        ) 

            task_tokens = context.new().resize_(context.size(0), 1).fill_(19) 

            
            # Move to GPU, if available
            context = context.to(args.device)
            features = features.to(args.device)
            spatials = spatials.to(args.device)
            segment_ids = segment_ids.to(args.device)
            input_mask = input_mask.to(args.device)
            image_mask = image_mask.to(args.device)
            co_attention_mask = co_attention_mask.to(args.device)
            task_tokens = task_tokens.to(args.device)

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
            
            pooled_output = pooled_output.to(args.device)
            caption = caption.to(args.device)
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

            loss = criterion(scores_packed, targets_packed)

            # Add doubly stochastic attention regularization
            loss += args.alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()

            # Keep track of metrics
            losses.update(loss.item(), sum(decode_lengths))
            top5 = accuracy(scores_packed, targets_packed, 5)
            top5accs.update(top5, sum(decode_lengths)) 
            batch_time.update(time.time() - start)

            start = time.time()

            if i % args.print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss_val:.4f} ({loss_avg:.4f})\t'
                      'Top-5 Accuracy {top5_val:.3f} ({top5_avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,                                                          loss_val = float(losses.val), loss_avg = float(losses.avg), top5_val = float(top5accs.val), top5_avg=float(top5accs.avg)))    
            
             # References
            for j in range(targets.shape[0]):
                img_caps = targets[j].tolist() # validation dataset only has 1 unique caption per img
                img_caps = tokenizer.convert_ids_to_tokens(img_caps) # th) it has to be a one sentence
                clean_cap = [w for w in img_caps if w not in ["[PAD]","[CLS]","[SEP]"]]  # remove pad, start, and end # clean function
                references.append(clean_cap) 
            
            # hypothesis
            # preds.shape torch.Size([32, 29])
            _, preds = torch.max(scores, dim=2)
                # _, preds ==> values, and index respectively 
                # "dim =1" means it extracts 928 elements from 928 * 30522. that is, criterion is dim 1 which will be shrinked
            # print(f"predicted logits : {preds}")
            preds = preds.tolist() 
            preds_token = []
            for l in preds : 
                preds_token.append(tokenizer.convert_ids_to_tokens(l))
            
            for j, p in enumerate(preds_token):
                # print(f"iter : {j}, p in preds : {p}, p shape : {len(p)}")
                # print(f"decode_lengths : {decode_lengths}, len : {len(decode_lengths)}")
                # pred = p[:decode_lengths[j]] # decode_lenths is from decoder's 3rd output, like ... 29? 30? 
                pred = p[:decode_lengths[j]]
                pred = [w for w in pred if w not in ["[PAD]", "[CLS]","[SEP]"]]
                hypothesis.append(pred)  # remove pads, start, and end

            assert len(references) == len(hypothesis)
        
        # Calculate BLEU-4 scores
        bleu4 = corpus_bleu(references, hypothesis)
        
        print(
            '\n * LOSS - {loss_avg:.3f}, TOP-5 ACCURACY - {top5_avg:.3f}, BLEU-4 - {bleu}\n'.format(
                loss_avg=float(losses.avg),
                top5_avg=float(top5accs.avg),
                bleu=bleu4))
        for r,h in zip(references,hypothesis):
            logger.info(' '.join(r)+ '\n' + ' '.join(h) + '\n')
        logger.info("*****Validation Done*****")
    return bleu4

if __name__ == '__main__':
    main()
