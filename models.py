import torch
from torch import nn
import torchvision
# from pytorch_transformers.tokenization_bert import BertTokenizer, BertModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """

        # encoder_out = encoder_out.repeat(1, 1024, 1)
        # print(f"I extend num pixels which means number of encoder hidden vectors : {encoder_out.shape}")

        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)    @@8, 1024, 512
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)             @@8, 512

        # ---in the attention function
        # att1 shape : torch.Size([8, 1, 512])
        # att2 shape : torch.Size([8, 512])
        # att2 unsqueeze(1) shape : torch.Size([8, 1, 512])


        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        # att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(1)  # (batch_size, attention_dim)
        # print(f"att shape : {att.shape}") # 8, 1024 # att shape : torch.Size([8, 1024])

        alpha = self.softmax(att)  # (batch_size, num_pixels) -> # (batch_size, attention_dim)
        # print(f"alpha shape : {alpha.shape}") # alpha : attention score 

        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        # print(f"att_wei_enc shape : {attention_weighted_encoding.shape}") # att_wei_enc shape : torch.Size([8, 1024])
        # att shape : torch.Size([8, 1])
        # alpha shape : torch.Size([8, 1])
        # att_wei_enc shape : torch.Size([8, 1024]) 
        # the shape is same !

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=768, dropout=0.5):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()

        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.embedding = nn.Embedding(vocab_size, embed_dim)  # leanable embedding layer
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        # print(f"### this print is from def init_hidden_state")
        # print(f"### to check encoder out size")
        # print(f"### encoder shape : {encoder_out.shape}")
        ### encoder shape : torch.Size([8, 1, 1024])

        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # batch size : 8 
        # encoder_dim : 1024 
        # encoder_out.shape : torch.Size([8, 1024])

        # Flatten image
        # encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim) # (batch_size, num_pixels, encoder_dim)
        # print(f"check the encoder shape ; {encoder_out.shape}") # check the encoder shape ; torch.Size([8, 1024, 1024])

        num_pixels = encoder_out.size(1)

        # after flatten, encoder shape : torch.Size([8, 1, 1024])
        # WTF num pixels : 1

        # print("--- in the forward ---")
        # print(f"after flatten, encoder shape : {encoder_out.shape}")
        # print(f"WTF num pixels : {num_pixels}")

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        # print(f"embeddings shape : {embeddings.shape}") embeddings shape : torch.Size([8, 30, 512])

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # decode_lengths = caption_lengths.tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds # [batch_size, t-th, vocab_size(30522)]
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas #, sort_ind

class DecoderWithBertEmbedding(nn.Module):

    def __init__(self, vocab_size, use_glove, use_bert, tokenizer, BertModel):
        super(DecoderWithBertEmbedding, self).__init__()
        self.encoder_dim = 768 #1024 #2048
        self.attention_dim = 512
        self.use_bert = use_bert
        self.tokenizer = tokenizer 
        self.bertmodel = BertModel

        if use_glove:
            self.embed_dim = 300
        elif use_bert:
            self.embed_dim = 768
        else:
            self.embed_dim = 512

        self.decoder_dim = 512
        self.vocab_size = vocab_size
        self.dropout = 0.5
        
        # soft attention
        self.enc_att = nn.Linear(1024, 512)
        self.dec_att = nn.Linear(512, 512)
        self.att = nn.Linear(512, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

        # decoder layers
        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(self.embed_dim + self.encoder_dim, self.decoder_dim, bias=True)
        self.h_lin = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.c_lin = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.f_beta = nn.Linear(self.decoder_dim, self.encoder_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.decoder_dim, self.vocab_size)

        # init variables
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        
        if not use_bert:
            self.embedding = nn.Embedding(vocab_size, self.embed_dim)
            self.embedding.weight.data.uniform_(-0.1, 0.1)

            # load Glove embeddings
            if use_glove:
                self.embedding.weight = nn.Parameter(glove_vectors)

            # always fine-tune embeddings (even with GloVe)
            for p in self.embedding.parameters():
                p.requires_grad = True

    def forward(self, encoder_out, encoded_captions, caption_lengths):    
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        dec_len = [x-1 for x in caption_lengths]
        max_dec_len = max(dec_len)
        # print(f"decoder.forward() -> max_dec_len : {max_dec_len}")  #tensor([29])

        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)
        num_pixels = encoder_out.size(1)

        # load bert or regular embeddings
        if not self.use_bert:
            embeddings = self.embedding(encoded_captions)
        elif self.use_bert:
            embeddings = []
            # print(f"Number of encoded captions : {len(encoded_captions)}") # Number of encoded captions : 32
            for cap_idx in  encoded_captions:
                cap_embedding_ = []
                
                """
                this part is .. 
                    data set에서 vocab의 word embedding을 가져와서, 그걸 다시 token으로 바꾼 다음, 
                    BERT embedding으로 바꿔서 그걸 self.bertmodel 에 넣어서 encoded_layer를 뽑아내고 있음. 
                    그러나 그럴 필요없는게, 우리는 바로 Bert embedding을 가지고 있으니 바로 넣어주면 됨. 
                """ 
                
                # print(f"cap_idx: {cap_idx}") # each caption 
    #             #  tensor([  101,  7908,  2012,  1996,  6765,  1997,  4302, 10693,  1998,  1996,
    #      2431,  1011,  2668,  3159,  1999,  2251,  2268,   102,     0,     0,
    #         0,     0,     0,     0,     0,     0,     0,     0,     0,     0],
    #   device='cuda:0')
                
                # 1. length  2. cap is.. str 
                
                # [SEP] token -> [PAD] token 
                for i in range(len(cap_idx)):
                    if cap_idx[i] == 102 : 
                        cap_idx[i] = 0  
                
                # decode 
                cap = self.tokenizer.decode(cap_idx.tolist()) # decoded sentence 
                
                # length alignment 
                try : 
                    assert len(cap.split()) == max_dec_len 
                except : 
                    while len(cap.split()) < max_dec_len: 
                        cap += ' [PAD]'
                        
                    while len(cap.split()) > max_dec_len : 
                          cap_ = cap.split()[:-1]
                          cap = " ".join(w for w in cap_)
                          
                    # print(f"cap len : {len(cap.split())}")  # 29 ( = max_len -1 = 30-1)
                
                # print(f"cap : {cap}")
                # cap : [CLS] the anasazi ruins were filmed at anza - borrego desert state park. [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD] [PAD]
#
                
                tokenized_cap = self.tokenizer.tokenize(cap)                
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_cap) # but it is same with encoded caption .. == cap_idx .. 
                tokens_tensor = torch.tensor(indexed_tokens).to(device)
                # print(f"decoder.forward()>>> cap_idx : {cap_idx}")
                # print(f"cap_idx device : {cap_idx.device}")
                # print(f"cap_idx type : {type(cap_idx)}")
                # print(f"cap_idx shape : {cap_idx.shape}")
                
                
                tokens_tensor = tokens_tensor.unsqueeze(-1)
                # cap_idx = cap_idx.unsqueeze(-1) # # # # 
                # print(f"after unsqueeze(-1) :{cap_idx.shape}")
                # print(f"tokens_tensor : {tokens_tensor}")
                with torch.no_grad():
                    encoded_layers, _ = self.bertmodel(tokens_tensor)
                    
                bert_embedding = encoded_layers[11].squeeze(0) #[1,768] -> #768
                # print("Bert Embedding", bert_embedding.shape) #[768]
                
                split_cap = cap.split()
                tokens_embedding = []
                j = 0

                for full_token in split_cap:
                    # print(f"full_token : {full_token}")
                    # full token is each token 
                            # "I", "like" , "apple" "[PAD]" , ... like this 
                    curr_token = ''
                    x = 0
                    for i,_ in enumerate(tokenized_cap[1:]): # disregard CLS
                        token = tokenized_cap[i+j]
                        piece_embedding = bert_embedding[i+j] #0차원 -> 1차원  # nothing in inside!!!!!!!!!!!!!!
                        # print(f"piece_embedding value : {piece_embedding}") # a float value like 0.07680762559175491
                        # print(f"token(=tokenized_cap[i+j]) : {token}") # same with full_token
                        
                        # full token
                        if token == full_token and curr_token == '' : # ., 이런기호들이 없으면 여기로 들어옴 
                            tokens_embedding.append(piece_embedding) # pice_embedding 은 하나의 상수값 
                            j += 1
                            break
                        else: # partial token # curr_token은 변경한 적이 없으니까 token!=full_token인 경우 말하는거 / . , 같은 것들이 가끔 token 으로 나뉘어짐
                            x += 1
                            if curr_token == '':
                                tokens_embedding.append(piece_embedding)
                                curr_token += token.replace('#', '')
                            else:
                                tokens_embedding[-1] = torch.add(tokens_embedding[-1], piece_embedding)
                                curr_token += token.replace('#', '')
                                
                                if curr_token == full_token: # end of partial
                                    j += x
                                    break
                        # print(f"tokens_embedding working ... ;;; {tokens_embedding}")

                # print("Token Embedding",tokens_embedding) # [tensor(), tensor(), ... , tensor()] # 29 # it is a list
                cap_embedding = torch.stack(tokens_embedding) 
                # just [29] / not stacking I think (or just convert to tensor from list)
                # cap_embedding = torch.stack(torch.Tensor([tokens_embedding,bert_embedding])) #[29] -> [29,768]
                # print("Caption Embedding",cap_embedding.shape) # torch.Size([29])
                # embeddings.append(cap_embedding) #[32,29] -> [32,29,768]
                embeddings.append(bert_embedding) # [32, 768] <- [32, 29]  # .. love .. ->..  token 1 .. 
  
            embeddings = torch.stack(embeddings)
            # print(f"embeddings shape : {embeddings.shape}") # torch.Size([32, 768])

        # init hidden state
        avg_enc_out = encoder_out.mean(dim=1)
        h = self.h_lin(avg_enc_out)
        c = self.c_lin(avg_enc_out)

        predictions = torch.zeros(batch_size, max_dec_len, vocab_size).to(device)
        alphas = torch.zeros(batch_size, max_dec_len, num_pixels).to(device)

        # print(f"dec_len ?? : {dec_len}") # 29
        # 0, 1, 2, 3, ,,, 29 (till max_dec_len)
        for t in range(max(dec_len)):
            # print(f"t ! in forward : {t}")
            batch_size_t = sum([l > t for l in dec_len ])
            
            # soft-attention
            enc_att = self.enc_att(encoder_out[:batch_size_t])
            dec_att = self.dec_att(h[:batch_size_t])
            att = self.att(self.relu(enc_att + dec_att.unsqueeze(1))).squeeze(2)
            alpha = self.softmax(att)
            attention_weighted_encoding = (encoder_out[:batch_size_t] * alpha.unsqueeze(2)).sum(dim=1)
        
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            attention_weighted_encoding = gate * attention_weighted_encoding # 8, 1024
            # print(f"attention_weighted_encoding shape : {attention_weighted_encoding.shape}") 
            # attention_weighted_encoding shape : torch.Size([32, 1024])
            
            # print(embeddings.shape) # torch.Size([32, 768])
            # print("Embeddings:",embeddings)  
            # print("Indexing is..",batch_size_t,t) #tensor([32]),  0th, 1st, 2nd, 3rd .. 
            batch_embeds = embeddings[:batch_size_t, :]       # : batch_size_t , t, :      
            cat_val = torch.cat([batch_embeds.double(), attention_weighted_encoding.double()], dim=1) # just attach them sequentially
            #concat val : [ batch_embeds + attention_weighted_encoding ]
                # 'batch_embeds' includes encoded caption information? as a input of each LSTM cell ????
            
            h, c = self.decode_step(cat_val.float(),(h[:batch_size_t].float(), c[:batch_size_t].float())) # decoding LSTMCell
            # RuntimeError: input has inconsistent input_size: got 1053(=1024+29) expected 1792(=1024 + 768)
            # h : tensor containing the next hidden state for each element in the batch
            # c : tensor containing the next cell state for each element in the batch
            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
            
        # preds, sorted capts, dec lens, attention wieghts
        return predictions, encoded_captions, dec_len, alphas