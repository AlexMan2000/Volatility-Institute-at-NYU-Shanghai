import torch
import torch.nn as nn
import torch.nn.functional as F
import lib


class Attention(nn.Module):
    """
    This class established the Attention Mechanism in the encoded image, also used for visualization.
    """
    def __init__(self,encoder_dim,decoder_dim,attention_dim=None,attention_method="concat",attention_type="global"):
        super().__init__()
        assert attention_type in ["global","local"]
        assert attention_method in ["dot","general","concat"], "method error"

        self.method = attention_method
        self.type = attention_type
        self.attention_dim = attention_dim

        if self.type == "local":
            self.attention_dim = attention_dim
            self.Wa = nn.Linear(decoder_dim,attention_dim)
            self.Va = nn.Linear(encoder_dim,attention_dim)
            self.out = nn.Linear(attention_dim,1)

        else:
            if self.method == "general":
                self.Wa = nn.Linear(encoder_dim,decoder_dim,bias=False)
            elif self.method == "concat":
                self.Wa = nn.Linear(encoder_dim+decoder_dim,decoder_dim,bias=False)
                self.Va = nn.Linear(decoder_dim,1)


    def forward(self,hidden_state,encoder_outputs):
        """
        This function computes the attention_weights used for decoder RNN caption generation.
        :param hidden_state: input hidden_state from the last LSTM cell, [num_layer,batch_size,decoder_dim]
        :param encoder_outputs: outputs of the CNN encoder. [batch_size,seq_len,encoder_hidden_size]
        :return:
        """
        if self.method == "dot":
            return self.dot_score(hidden_state,encoder_outputs)

        elif self.method == "general":
            return self.general_score(hidden_state,encoder_outputs)

        elif self.method == "concat":
            return self.concat_score(hidden_state,encoder_outputs)


    def dot_score(self,hidden_state,encoder_outputs):
        """
        Depreciated, won't be used for the project
        :param hidden_state: [batch_size,decoder_dim]
        :param encoder_outputs: [batch_size,seq_len,encoder_dim]
        :return:
        """
        hidden_state = hidden_state.permute(1, 2, 0)  # [batch_size,decoder_dim,1]
        attention_weight = encoder_outputs.bmm(hidden_state).squeeze(-1)  # [batch_size,seq_len]
        attention_weight = F.softmax(attention_weight)
        context = encoder_outputs * attention_weight.unsqueeze(2)
        context = context.sum(dim=1)  # [batch_size,encoder_dim]
        return attention_weight,context


    def general_score(self,hidden_state,encoder_outputs):
        """
        Depreciated, won't be used for the project.
        :param hidden_state: [batch_size,decoder_dim]
        :param encoder_outputs: [batch_size,seq_len,encoder_dim]
        :return:
        """
        batch_size = encoder_outputs.size(0)
        encoder_seq_len = encoder_outputs.size(1)
        print(encoder_outputs.size())
        print(encoder_outputs.view(batch_size*encoder_seq_len,-1))

        encoder_outputs = self.Wa(encoder_outputs.view(batch_size*encoder_seq_len,-1))  # [batch_size*seq_len,decoder_dim]
        encoder_outputs = encoder_outputs.view(batch_size,encoder_seq_len,-1)  # [batch_size,seq_len,decoder_dim]
        hidden_state = hidden_state.permute(1, 2, 0)  # [batch_size,decoder_dim,1]
        attention_weight = encoder_outputs.bmm(hidden_state).squeeze(-1)  # [batch_size,seq_len]
        attention_weight = F.softmax(attention_weight)
        context = encoder_outputs * attention_weight.unsqueeze(2)
        context = context.sum(dim=1)  # [batch_size,encoder_dim]
        return attention_weight,context


    def concat_score(self,hidden_state,encoder_outputs):
        """
        Defining the alignment function mentioned by Luong et al.
        :param hidden_state: [batch_size,decoder_dim]
        :param encoder_outputs: [batch_size,seq_len,encoder_dim]
        :return:
        """
        # If we use the local attention
        if self.type=="local":
            encoder_out = self.Va(encoder_outputs) #[batch_size,seq_len,attention_dim]
            decoder_out = self.Wa(hidden_state) #[batch_size,attention_dim]

            combined_states = torch.tanh(encoder_out+decoder_out.unsqueeze(1))

            attention_scores = self.out(combined_states)

            attention_scores = attention_scores.squeeze(2)

            attention_weight = F.softmax(attention_scores,dim=1)

            context = encoder_out * attention_weight.unsqueeze(2)

            context = context.sum(dim=1)
        # If we use the global attention
        else:
            hidden_state = hidden_state.unsqueeze(1)
            hidden_state = hidden_state.repeat(1, encoder_outputs.size(1), 1)  # [batch_size,seq_len,decoder_dim]

            concated = torch.cat([hidden_state, encoder_outputs],
                                 dim=-1)  # [batch_size,seq_len,decoder_dim+encoder_dim]

            batch_size = encoder_outputs.size(0)
            encoder_seq_len = encoder_outputs.size(1)

            attention_weight = self.Va(torch.tanh(self.Wa(concated.view(batch_size*encoder_seq_len,-1)))).squeeze(-1)  # [batch_size*seq_len]
            attention_weight = attention_weight.view(batch_size,encoder_seq_len)

            attention_weight = F.softmax(attention_weight,dim=1)  # [batch_size,seq_len]
            context = encoder_outputs * attention_weight.unsqueeze(2) #[batch_size,seq_len,encoder_dim]
            context = context.sum(dim=1) #[batch_size,encoder_dim]
        return attention_weight,context