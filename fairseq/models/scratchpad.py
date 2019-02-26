### ENCODER ###
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from fairseq import utils
from fairseq.models import FairseqEncoder
from fairseq.models.lstm import LSTMEncoder

class BADLSTMEncoder(FairseqEncoder):

  def __init__(
    self, args, dictionary, embed_dim=128, hidden_dim=512, n_layers=2, dropout=0.1
  ):
    super().__init__(dictionary)
    self.args = args
    self.n_layers = n_layers
    self.h_dim = hidden_dim // 2

    self.embed_tokens = nn.Embedding(
      num_embeddings = len(dictionary),
      embedding_dim = embed_dim,
      padding_idx = dictionary.pad(),
    )

    self.lstm = nn.LSTM(
      input_size = embed_dim,
      hidden_size=self.h_dim,
      num_layers=n_layers,
      dropout=dropout,
      bidirectional=True
    )

  def get_pad(self, src_lengths, ssize):
    idxs = torch.arange(ssize[0]).expand(ssize[1], ssize[0]).t().to(src_lengths.device)
    return (idxs < src_lengths.unsqueeze(1)).t() # BxT

  def forward(self, src_tokens, src_lengths):
    bs = src_lengths.size(0)
    pad_mask = self.get_pad(src_lengths, src_tokens.size())

    x = self.embed_tokens(src_tokens)
    x = pack_padded_sequence(x, src_lengths, batch_first=True)
    outputs, (h_n, c_n) = self.lstm(x)
    unpacked_outputs, _ = pad_packed_sequence(outputs, batch_first=False)
    h_n_b_l = h_n.view(
      self.n_layers, 2, bs, self.h_dim
    ).transpose(
      1, 2
    ).contiguous().view(
      self.n_layers, bs, self.h_dim * 2
    )
    return {
      'output': unpacked_outputs,
      'final': h_n_b_l,
      'pad_mask': pad_mask,
      'lens': src_lengths,
    }

  def reorder_encoder_out(self, encoder_out, new_order):
    return {
      'output': encoder_out['output'].index_select(new_order, dim=1),
      'pad_mask': encoder_out['pad_mask'].index_select(new_order, dim=0),
      'final': encoder_out['final'].index_select(new_order, dim=1),
      'lens': encouder_out['lens'][new_order],
    }
from fairseq.models import FairseqModel, register_model
#################


### DECODER ###
import torch
from torch.nn import functional as F

from fairseq.models import FairseqIncrementalDecoder

class MultiLayerGRUCell(nn.Module):

  def __init__(self, embed_size, hidden_size, n_layers, drop_prob):
    super().__init__()
    self.n_layers = n_layers
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.n_layers = n_layers
    self.rnns = nn.ModuleList(
      [nn.GRUCell(
        input_size = self.embed_size, #+ self.hidden_size,
        hidden_size = self.hidden_size
      )] + \
      [nn.GRUCell(
         input_size = self.hidden_size,
         hidden_size = self.hidden_size,
       ) for i in range(self.n_layers - 1)]
    )
    self.dropout = nn.Dropout(drop_prob)


  def forward(self, x, h):
    x_l = x
    new_h = []

    for i, (rnn, h_l) in enumerate(zip(self.rnns, h)):
      x_n = rnn(self.dropout(x_l), h_l)
      new_h.append(x_n)
      if (i == 0):
        x_l = x_n
      else:
        x_l = x_n + x_l
    return torch.stack(new_h, 0)


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
                # recurrent cell
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
        def reorder_state(state, idx=0):
            if isinstance(state, list) or isinstance(state, tuple):
                return [reorder_state(state_i, idx) for (state_i, idx) in zip(state, [0, 0, 0, 1])]
            return state.index_select(idx, new_order)

        #new_state = tuple(map(reorder_state, cached_state))
        new_state = reorder_state(cached_state)
        utils.set_incremental_state(self, incremental_state, 'cached_state', new_state)

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return int(1e5)  # an arbitrary large number

    def make_generation_fast_(self, need_attn=False, **kwargs):
        self.need_attn = need_attn

class ScratchpadDecoder(FairseqIncrementalDecoder):

  def __init__(
    self, dictionary, encoder_hidden_dim=512, embed_dim=512, hidden_dim=512,
    n_layers=1, dropout=0.1
  ):
    super().__init__(dictionary)
    self.n_layers = n_layers

    self.embed_tokens = nn.Embedding(
      num_embeddings = len(dictionary),
      embedding_dim=embed_dim,
      padding_idx=dictionary.pad(),
    )

    self.enc2dec = nn.Sequential(
      nn.Linear(encoder_hidden_dim, hidden_dim),
      nn.Tanh()
    )


#    self.attention = Attention(
#      hidden_dim, hidden_dim, hidden_dim,
#    )
    self.attention = AttentionLayer(
      encoder_hidden_dim, hidden_dim, encoder_hidden_dim
    )

    self.rnn = MultiLayerGRUCell(
      embed_size = embed_dim + encoder_hidden_dim,
      hidden_size = hidden_dim,
      n_layers = n_layers,
      drop_prob = dropout,
    )
#    self.rnn = nn.GRUCell(
#      embed_dim + encoder_hidden_dim,
#      hidden_dim
#    )

    self.out = nn.Sequential(
      nn.Linear(hidden_dim + encoder_hidden_dim, len(dictionary)),
#      nn.Linear(hidden_dim, len(dictionary)),
    )

    self.attentive_writer = AttentiveWriter(
      encoder_hidden_dim + hidden_dim, encoder_hidden_dim, encoder_hidden_dim
    )

  def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
    if incremental_state is None:
      max_len = prev_output_tokens.size(1)
      res_out = []
      attn_out = []
      incremental_state = {}
      for i in range(max_len):
#        import pdb; pdb.set_trace()
        res, attn = self.forward(prev_output_tokens, encoder_out, incremental_state)
        res_out.append(res)
        attn_out.append(attn)
#      tbh = torch.stack(res_out, 0)
#      packed = pack_padded_sequence(tbh, encoder_out['lens'])
#      return packed
#      import pdb; pdb.set_trace()
#      return torch.cat(res_out, 0), None
      tbh = torch.stack(res_out, 0).view(-1, res_out[0].size(1))
      abh = torch.stack(attn_out, 1)
      return tbh, abh
    elif len(incremental_state) == 0: # incremental_state == {}
      tgt_idx = 0
      enc_out, final, _ = encoder_out['encoder_out']
      hidden = self.enc2dec(final[-self.n_layers:])
#      hidden = self.enc2dec(encoder_out['final'])[-self.n_layers:]
    else:
      tgt_idx = utils.get_incremental_state(
        self, incremental_state, 'tgt_idx') + 1
      enc_out = utils.get_incremental_state(
        self, incremental_state, 'enc_out')
      hidden = utils.get_incremental_state(
        self, incremental_state, 'hidden')

    pad_mask = encoder_out['encoder_padding_mask']
    feed_tok = prev_output_tokens[:, tgt_idx]
    last_hidden = hidden[-1]

    embedded = self.embed_tokens(feed_tok)
    context, attn = self.attention(
      last_hidden, enc_out, pad_mask
    )
    rnn_input = torch.cat([embedded, context], 1)
    hidden = self.rnn(rnn_input, hidden)
#    import pdb; pdb.set_trace()
    with_context = torch.cat([hidden[-1], context], 1)
#    with_context = hidden[-1]
    output = self.out(with_context)

#    enc_out = self.attentive_writer(with_context, enc_out, pad_mask=pad_mask)

    # set the incremental state
    utils.set_incremental_state(
      self, incremental_state, 'enc_out', enc_out
    )
    utils.set_incremental_state(
      self, incremental_state, 'hidden', hidden
    )
    utils.set_incremental_state(
      self, incremental_state, 'tgt_idx', tgt_idx
    )
    return output, attn

  def reorder_incremental_state(self, incremental_state, new_order):
    enc_out = utils.get_incremental_state(
      self, incremental_state, 'enc_out'
    )
    hidden = utils.get_incremental_state(
      self, incremental_state, 'hidden'
    )

    utils.set_incremental_state(
      self, incremental_state, 'enc_out', enc_out.index_select(new_order, dim=1)
    )
    utils.set_incremental_state(
      self, incremental_state, 'hidden', hidden.index_select(new_order, dim=1)
    )

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
#    import pdb; pdb.set_trace()
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
      bidirectional = True
    )
    decoder = LSTMDecoder(
      dictionary = task.target_dictionary,
      embed_dim = args.decoder_embed_dim,
      hidden_size = args.decoder_hidden_dim,
      num_layers = args.decoder_n_layers,
      encoder_output_units = args.encoder_hidden_dim,
#      use_scratchpad = args.scratchpad,
      use_scratchpad=True,
#      use_scratchpad=False,
    )

    model = Scratchpad(encoder, decoder)
    print(model)

    return model

    # def forward(self, src_tokens, src_lengths, prev_output_tokens):
    #     encoder_out = self.encoder(src_tokens, src_lengths)
    #     decoder_out = self.decoder(prev_output_tokens, encoder_out)
    #     return decoder_out

#def forward(self, src_tokens, src_lengths, prev_output_tokens):



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
