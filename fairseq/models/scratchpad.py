### ENCODER ###
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from fairseq import utils
from fairseq.models import FairseqEncoder
from fairseq.models.lstm import LSTMEncoder

import numpy as np

from fairseq.models import FairseqModel, register_model
#################


### DECODER ###
import torch
from torch.nn import functional as F

from fairseq.models import FairseqIncrementalDecoder

from fairseq.models.lstm import Embedding, LSTMCell

class LSTMDecoder(FairseqIncrementalDecoder):
    """LSTM decoder."""
    def __init__(
        self, dictionary, embed_dim=512, hidden_size=512, out_embed_dim=512,
        num_layers=1, dropout_in=0.1, dropout_out=0.1, attention=True,
        encoder_output_units=512, pretrained_embed=None,
        share_input_output_embed=False, adaptive_softmax_cutoff=None,
        use_scratchpad = False,
    ):
        super().__init__(dictionary)
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.hidden_size = hidden_size
        self.share_input_output_embed = share_input_output_embed
        self.need_attn = True
        self.use_scratchpad = use_scratchpad

        self.adaptive_softmax = None
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        if pretrained_embed is None:
            self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        else:
            self.embed_tokens = pretrained_embed

        self.encoder_output_units = encoder_output_units
        if encoder_output_units != hidden_size:
            self.encoder_hidden_proj = Linear(encoder_output_units, hidden_size)
            self.encoder_cell_proj = Linear(encoder_output_units, hidden_size)
        else:
            self.encoder_hidden_proj = self.encoder_cell_proj = None
        self.layers = nn.ModuleList([
            LSTMCell(
                input_size=hidden_size + embed_dim if layer == 0 else hidden_size,
                hidden_size=hidden_size,
            )
            for layer in range(num_layers)
        ])
        if attention:
            # TODO make bias configurable
            self.attention = AttentionLayer(hidden_size, encoder_output_units, hidden_size, bias=False)
        else:
            self.attention = None
        if hidden_size != out_embed_dim:
            self.additional_fc = Linear(hidden_size, out_embed_dim)
        if adaptive_softmax_cutoff is not None:
            # setting adaptive_softmax dropout to dropout_out for now but can be redefined
            self.adaptive_softmax = AdaptiveSoftmax(num_embeddings, embed_dim, adaptive_softmax_cutoff,
                                                    dropout=dropout_out)
        elif not self.share_input_output_embed:
            self.fc_out = Linear(out_embed_dim, num_embeddings, dropout=dropout_out)
        #EDITED
        if self.use_scratchpad:
          self.attentive_writer = AttentiveWriter(
            hidden_size, encoder_output_units, encoder_output_units
          )

    def forward(self, prev_output_tokens, encoder_out_dict, incremental_state=None):
        encoder_out = encoder_out_dict['encoder_out']
        encoder_padding_mask = encoder_out_dict['encoder_padding_mask']

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
        bsz, seqlen = prev_output_tokens.size()

        # get outputs from encoder
        encoder_outs, encoder_hiddens, encoder_cells = encoder_out[:3]
        srclen = encoder_outs.size(0)

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)
        x = F.dropout(x, p=self.dropout_in, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # initialize previous states (or get from cache during incremental generation)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is not None:
            #EDITED
            prev_hiddens, prev_cells, input_feed, encoder_outs = cached_state
        else:
            num_layers = len(self.layers)
            prev_hiddens = [encoder_hiddens[i] for i in range(num_layers)]
            prev_cells = [encoder_cells[i] for i in range(num_layers)]
            if self.encoder_hidden_proj is not None:
                prev_hiddens = [self.encoder_hidden_proj(x) for x in prev_hiddens]
                prev_cells = [self.encoder_cell_proj(x) for x in prev_cells]
            input_feed = x.new_zeros(bsz, self.hidden_size)


        attn_scores = x.new_zeros(srclen, seqlen, bsz)
        outs = []
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            input = torch.cat((x[j, :, :], input_feed), dim=1)

            for i, rnn in enumerate(self.layers):
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout_out, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            if self.attention is not None:
#                import pdb; pdb.set_trace()
                out, attn_scores[:, j, :] = self.attention(hidden, encoder_outs, encoder_padding_mask)
            else:
                out = hidden
            out = F.dropout(out, p=self.dropout_out, training=self.training)

            # input feeding
            input_feed = out
            #EDITED
            if self.use_scratchpad:
              encoder_outs = self.attentive_writer(out, encoder_outs, encoder_padding_mask)

            # save final output
            outs.append(out)

        # cache previous states (no-op except during incremental generation)
        utils.set_incremental_state(
            self, incremental_state, 'cached_state',
            (prev_hiddens, prev_cells, input_feed, encoder_outs),
        )

        # collect outputs across time steps
        x = torch.cat(outs, dim=0).view(seqlen, bsz, self.hidden_size)

        # T x B x C -> B x T x C
        x = x.transpose(1, 0)

        # srclen x tgtlen x bsz -> bsz x tgtlen x srclen
        if not self.training and self.need_attn:
            attn_scores = attn_scores.transpose(0, 2)
        else:
            attn_scores = None

        # project back to size of vocabulary
        if self.adaptive_softmax is None:
            if hasattr(self, 'additional_fc'):
                x = self.additional_fc(x)
                x = F.dropout(x, p=self.dropout_out, training=self.training)
            if self.share_input_output_embed:
                x = F.linear(x, self.embed_tokens.weight)
            else:
                x = self.fc_out(x)
        return x, attn_scores

    def reorder_incremental_state(self, incremental_state, new_order):
        super().reorder_incremental_state(incremental_state, new_order)
        cached_state = utils.get_incremental_state(self, incremental_state, 'cached_state')
        if cached_state is None:
            return

        #EDITED
        def reorder_state(state, idx):
          if isinstance(state, list) or isinstance(state, tuple):
            return [reorder_state(state_i, idx) for state_i in state]
          return state.index_select(idx, new_order)

        new_state = [reorder_state(sub, idx) for (sub, idx) in zip(cached_state, [0, 0, 0, 1])]
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn


from fairseq.models.lstm import Linear
class AttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, source_embed_dim, output_embed_dim, bias=False):
        super().__init__()

        self.input_proj = Linear(input_embed_dim, source_embed_dim, bias=bias)
        self.output_proj = Linear(input_embed_dim + source_embed_dim, output_embed_dim, bias=bias)

    def forward(self, input, source_hids, encoder_padding_mask):
        # input: bsz x input_embed_dim
        # source_hids: srclen x bsz x output_embed_dim

        # x: bsz x output_embed_dim
        x = self.input_proj(input)

        # compute attention
        attn_scores = (source_hids * x.unsqueeze(0)).sum(dim=2)

        # don't attend over padding
        if encoder_padding_mask is not None:
            attn_scores = attn_scores.float().masked_fill_(
                encoder_padding_mask,
                float('-inf')
            ).type_as(attn_scores)  # FP16 support: cast to float and back
#              print("encoder_padding_mask: {0}".format(encoder_padding_mask.size()))
#          print("attn_scores: {0}".format(attn_scores.size()))
#          print("input: {0}".format(input.size()))
#          print("source_hids: {0}".format(source_hids.size()))
        attn_scores = F.softmax(attn_scores, dim=0)  # srclen x bsz

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=0)

        x = F.tanh(self.output_proj(torch.cat((x, input), dim=1)))
        return x, attn_scores

class Attention(nn.Module):
  def __init__(self, query_size, key_size, hidden_size):
    super().__init__()
    self.set_sizes(query_size, key_size, hidden_size)

#    self.q2h = nn.Sequential(
#      nn.Linear(self.q_size, self.h_size),
#    )
#
#    self.k2h = nn.Sequential(
#      nn.Linear(self.k_size, self.h_size),
#    )
#
#    self.to_score = nn.Sequential(
#      nn.Linear(self.h_size, 1, bias=False),
#    )
    self.to_score = nn.Sequential(
      nn.Linear(self.q_size + self.k_size, self.h_size, bias=False),
      nn.LeakyReLU(),
      nn.Linear(self.h_size, 1, bias=False),
    )

  def set_sizes(self, q, k, h):
    self.q_size = q
    self.k_size = k
    self.h_size = h

  def forward(self, query, keys, values = None, pad_mask = None):
    attn_energies = self.score(query, keys)
    probs = F.softmax(attn_energies, dim=1)
    if pad_mask is not None:
      probs = probs * pad_mask
      normalization_factor = probs.sum(1, keepdim=True)
      probs = probs / normalization_factor
    if values is not None:
      return probs.unsqueeze(1).bmm(values.transpose(0, 1)).squeeze(), probs
    return probs

  def score(self, query, keys):
    batch = query.size(0)
    timestep = keys.size(0)
#    q = F.tanh(self.q2h(query)).unsqueeze(0).expand(timestep, -1, -1)
#    k = F.tanh(self.k2h(keys))
#    energy = self.to_score(q * k).view(timestep, batch).t()
    cat = torch.cat([query.unsqueeze(0).expand(timestep, -1, -1), keys], 2)
    energy = self.to_score(cat).view(timestep, batch).t()
#    q = self.q2h(query).unsqueeze(0).expand(timestep, -1, -1)
#    k = self.k2h(keys)
#    energy = self.to_score(F.tanh(q + k)).squeeze().t()
#    energy = self.to_score(q*k).squeeze().t()
#    return energy.new(*energy.size()).fill_(1)
    return energy  # [B x T]

class AttentiveWriter(Attention):
  def __init__(self, query_size, key_value_size, hidden_size):
    super().__init__(query_size, key_value_size, hidden_size)
    self.to_write = nn.Sequential(
      nn.Linear(self.q_size, self.h_size),
      nn.ReLU(inplace=True),
#      nn.PReLU(self.h_size),
#      nn.LeakyReLU(),
      nn.Linear(self.h_size, self.k_size),
      nn.Tanh() # to bound updates in [-1, 1], since gru outputs are [-1, 1]
    )

  def forward(self, query, store, pad_mask = None):
    write_score = super().score(query, store)
    write_attn = F.sigmoid(write_score).unsqueeze(2).transpose(0, 1)
    if pad_mask is not None:
      try:
        write_attn = write_attn * (1 - pad_mask.float()).unsqueeze(2)
      except:
        import pdb; pdb.set_trace()
    update = self.to_write(query).unsqueeze(0)
    return (write_attn * update) + ((1 - write_attn) * store)

###########################################################



#from fairseq.models.lstm_encoder import LSTMEncoder
#from fairseq.models.scratchpad_decoder import ScratchpadDecoder
from fairseq.models import FairseqModel, register_model


@register_model('scratchpad')
class Scratchpad(FairseqModel):

  @staticmethod
  def add_args(parser):
    ###            Encoder Arguments              ###
    parser.add_argument(
      '--encoder-embed-dim',   type=int,   metavar='N')
    parser.add_argument(
      '--encoder-hidden-dim',  type=int,   metavar='N')
    parser.add_argument(
      '--encoder-n-layers',    type=int,   metavar='N')
    parser.add_argument(
      '--encoder-dropout',     type=float, metavar='N')
    #################################################

    ###            Decoder Arguments              ###
    parser.add_argument(
      '--decoder-embed-dim',   type=int,   metavar='N')
    parser.add_argument(
      '--decoder-hidden-dim',  type=int,   metavar='N')
    parser.add_argument(
      '--decoder-n-layers',    type=int,   metavar='N')
    parser.add_argument(
      '--decoder-dropout',     type=float, metavar='N')
    parser.add_argument(
      '--scratchpad', dest='scratchpad', action='store_true')
    parser.add_argument(
      '--no-scratchpad', dest='scratchpad', action='store_false')
    #################################################


  @classmethod
  def build_model(cls, args, task):
    encoder = LSTMEncoder(
      dictionary=task.source_dictionary,
      embed_dim = args.encoder_embed_dim,
      hidden_size = args.encoder_hidden_dim // 2,
      num_layers = args.encoder_n_layers,
      dropout_out = args.encoder_dropout,
      dropout_in = args.encoder_dropout,
      bidirectional = True
    )
    decoder = LSTMDecoder(
      dictionary = task.target_dictionary,
      embed_dim = args.decoder_embed_dim,
      hidden_size = args.decoder_hidden_dim,
      num_layers = args.decoder_n_layers,
      encoder_output_units = args.encoder_hidden_dim,
      dropout_out = args.decoder_dropout,
      dropout_in = args.decoder_dropout,
      use_scratchpad = args.scratchpad,
    )

    model = Scratchpad(encoder, decoder)
    print(model)
    n_par = sum([np.prod(p.size()) for p in model.parameters()])
    print("N model params!: {:,}".format(n_par))

    return model


from fairseq.models import register_model_architecture

@register_model_architecture('scratchpad', 'scratchpad_arch')
def scratchpad_arch(args):
  ###                      Encoder Arguments                      ###
  args.encoder_embed_dim =   getattr(args, 'encoder_embed_dim',  128)
  args.encoder_hidden_dim =  getattr(args, 'encoder_hidden_dim', 512)
  args.encoder_n_layers =    getattr(args, 'encoder_n_layers',     3)
  args.encoder_dropout =     getattr(args, 'encoder_dropout',    0.1)
  ###################################################################

  ###                      Decoder Arguments                      ###
  args.decoder_embed_dim =   getattr(args, 'decoder_embed_dim',  128)
  args.decoder_hidden_dim =  getattr(args, 'decoder_hidden_dim', 512)
  args.decoder_n_layers =    getattr(args, 'decoder_n_layers',     3)
  args.decoder_dropout =     getattr(args, 'decoder_dropout',    0.1)
  args.scratchpad =          getattr(args, 'scratchpad',        True)
  ###################################################################
