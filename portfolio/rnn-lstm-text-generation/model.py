import torch
from typing import Type
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from dataset import TextDataset


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        """
        YOUR CODE HERE (âŠƒï½¡â€¢Ìâ€¿â€¢Ì€ï½¡)âŠƒâ”âœ¿âœ¿âœ¿âœ¿âœ¿âœ¿
        Create necessary layers
        """
        self.embedding = nn.Embedding(self.vocab_size, embed_size, padding_idx=dataset.pad_id)
        self.rnn = rnn_type(input_size=embed_size, hidden_size=hidden_size, num_layers=rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        # This is a placeholder, you may remove it.
        #logits = torch.randn(
        #    indices.shape[0], indices.shape[1], self.vocab_size,
        #    device=indices.device)
        """
        YOUR CODE HERE (âŠƒï½¡â€¢Ìâ€¿â€¢Ì€ï½¡)âŠƒâ”âœ¿âœ¿âœ¿âœ¿âœ¿âœ¿
        Convert indices to embeddings, pass them through recurrent layers
        and apply output linear layer to obtain the logits
        """
        embs = self.embedding(indices)

        pack = pack_padded_sequence(embs, lengths.cpu(), batch_first=True, enforce_sorted=False)
        pack_out, _ = self.rnn(pack)

        out, _ = pad_packed_sequence(pack_out, batch_first=True)
        logits = self.linear(out)
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        # This is a placeholder, you may remove it.
        #generated = prefix + ', Ð° Ð¿Ð¾Ñ‚Ð¾Ð¼ ÐºÑƒÐ¿Ð¸Ð» Ð¼ÑƒÐ¶Ð¸Ðº ÑˆÐ»ÑÐ¿Ñƒ, Ð° Ð¾Ð½Ð° ÐµÐ¼Ñƒ ÐºÐ°Ðº Ñ€Ð°Ð·.'
        """
        YOUR CODE HERE (âŠƒï½¡â€¢Ìâ€¿â€¢Ì€ï½¡)âŠƒâ”âœ¿âœ¿âœ¿âœ¿âœ¿âœ¿
        Encode the prefix (do not forget the BOS token!),
        pass it through the model to accumulate RNN hidden state and
        generate new tokens sequentially, sampling from categorical distribution,
        until EOS token or reaching self.max_length.
        Do not forget to divide predicted logits by temperature before sampling
        """
        dev = next(self.parameters()).device
        temp = max(float(temp), 1e-6)

        prefix_id = self.dataset.text2ids(prefix) if prefix else []
        gen_ids = [self.dataset.bos_id] + prefix_id

        input_ids = torch.tensor(gen_ids, dtype=torch.long, device=dev).unsqueeze(0)
        emb = self.embedding(input_ids)
        _, hidden = self.rnn(emb)

        cur_t = input_ids[:, -1:]

        for _ in range(self.max_length - len(gen_ids)):
            cur_emb = self.embedding(cur_t)
            output, hidden = self.rnn(cur_emb, hidden)
            logits = self.linear(output[:, -1, :]) / temp
            probs = torch.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)

            tok_id = next_t.item()
            if tok_id == self.dataset.eos_id:
                break

            gen_ids.append(tok_id)
            cur_t = next_t

        clean_ids = [
            idx for idx in gen_ids
            if idx not in {self.dataset.bos_id, self.dataset.eos_id, self.dataset.pad_id}
        ]
        generated = prefix + self.dataset.ids2text(clean_ids)
        return generated