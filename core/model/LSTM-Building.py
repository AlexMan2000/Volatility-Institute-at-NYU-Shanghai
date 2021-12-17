import torch


class MyDecoderRNN(nn.Module):
    """
    This class Establish an attention-based Rnn Module for caption generation
    """
    def __init__(self,vocab_dim,
                 embedding_dim,
                 encoder_dim,
                 decoder_dim,
                 n_layers = 1,
                 dropout_p=0.3,
                 attention_dim=None,
                 attention_type = "global",
                 embedding_type="randomized",
                 GRU = False):
        super().__init__()

        # 定义参数
        self.attention_type = attention_type
        self.vocab_dim = vocab_dim
        self.n_layers = n_layers
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.attention_dim = attention_dim
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        self.GRU = GRU

        assert attention_type in ["local","global","none"],"Expected local or global, got {} instead.".format(attention_type)

        assert embedding_type in ["randomized","Glove"],"Expected randomized,Glove, got {} instead.".format(embedding_type)

        # Without Attention
        if self.attention_type == "none":
            self.lstm = nn.LSTM(self.embedding_dim,decoder_dim,num_layers=self.n_layers,batch_first=True)


        # 定义embed层，将输入的caption 经过embed处理之后用于decoder生成image captions
        if self.embedding_type == "randomized":
            self.embedding_layer = nn.Embedding(num_embeddings=VOCAB_DIM,
                                                embedding_dim=self.embedding_dim,
                                                padding_idx=PAD_IDX)
        elif self.embedding_type == "Glove":
            from GloveEmbedding import Glove_embedding_matrix
            self.embedding_dim += 4
            self.embedding_layer = nn.Embedding(num_embeddings=VOCAB_DIM,
                                                embedding_dim=self.embedding_dim,
                                                padding_idx=PAD_IDX)


            self.embedding_layer.weight = nn.Parameter(torch.tensor(Glove_embedding_matrix,dtype=torch.float32))
            self.embedding_layer.weight.requires_grad = False


        # 定义attention层，产生attention_score 用于后续分配注意力
        if self.attention_type == "global":
            self.attn = Attention(encoder_dim,decoder_dim)

        elif self.attention_type == "local":
            self.attn = Attention(encoder_dim,decoder_dim,attention_dim=self.attention_dim,attention_type="local")

        # 定义正则化方法
        self.dropout = nn.Dropout(dropout_p)

        # 定义初始化层
        self.init_hidden = nn.Linear(encoder_dim,decoder_dim)
        self.init_cell = nn.Linear(encoder_dim,decoder_dim)

        # 定义每一个LSTM cell单元用于手动迭代
        if not self.GRU:
            if self.attention_type=="local":
                self.lstm_cell = nn.LSTMCell(self.embedding_dim+attention_dim,decoder_dim,bias=True)
            else:
                self.lstm_cell = nn.LSTMCell(self.embedding_dim+encoder_dim,decoder_dim,bias=True)

        elif self.GRU:
            if self.attention_type == "local":
                self.gru_cell = nn.GRUCell(self.embedding_dim+attention_dim,decoder_dim,bias=True)
            else:
                self.gru_cell = nn.GRUCell(self.embedding_dim+encoder_dim,decoder_dim,bias=True)


        self.fcn = nn.Linear(decoder_dim,vocab_dim)


    def forward(self,image_features,captions):
        """
        :param image_features: encoder_outputs [batch_size,seq_len,encoder_dim]
        :param captions: numericalized captions list  [batch_size,max_len]
        :return:
        """
        if self.attention_type=="none":
            embedded_captions = self.embedding_layer(captions[:,:-1])
            #concat the features and captions
            x = torch.cat((image_features.unsqueeze(1),embedded_captions),dim=1)
            x,_ = self.lstm(x)
            x = self.fcn(x)
            return (x,None)


        else:
            embedded_captions = self.embedding_layer(captions) #[batch_size,embed_dim]

            # 初始化LSTM层
            # 对所有的features取平均用于初始化hidden_state和cell_state
            image_features_init = image_features.mean(dim=1)


            hidden_state = self.init_hidden(image_features_init)
            cell = self.init_cell(image_features_init)

            # 遍历所有时间步
            seq_len = len(captions[0])-1
            batch_size = captions.size(0)
            encoder_dim = image_features.size(1)

            # 初始化一个batch_size的所有的结果
            outputs = torch.zeros(batch_size,seq_len,self.vocab_dim).to(lib.DEVICE)
            attention_weights = torch.zeros(batch_size,seq_len,encoder_dim).to(lib.DEVICE)

            if self.GRU:
                for t in range(seq_len):
                    attention_weight, context = self.attn(hidden_state, image_features)

                    gru_input = torch.cat([embedded_captions[:, t], context], dim=1)

                    hidden_state = self.gru_cell(gru_input, hidden_state)

                    output = self.fcn(self.dropout(hidden_state))

                    # 预测的词向量, output [batch_size,vocab_dim] ,attention_weight [batch_size,seq_len]
                    outputs[:, t] = output
                    attention_weights[:, t] = attention_weight

            else:
            #对于每一个lstm cell 我们都需要输入四个数据，hidden_state,cell,上一次 attention产生的context, 以及上一次的output(embedded之后的)
                for t in range(seq_len):

                    attention_weight,context = self.attn(hidden_state,image_features)
                    lstm_input = torch.cat([embedded_captions[:,t],context],dim=1)
                    hidden_state, cell = self.lstm_cell(lstm_input,(hidden_state,cell))

                    output = self.fcn(self.dropout(hidden_state))

                    #预测的词向量, output [batch_size,vocab_dim] ,attention_weight [batch_size,seq_len]
                    outputs[:,t] = output
                    attention_weights[:,t] = attention_weight

            return outputs,attention_weights
