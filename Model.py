from __future__ import division, print_function
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Model(nn.Module):
    """
    Minimalistic PyTorch model for HTR - Support Devanagari Language
    """

    # model constants
    batchSize = 32
    img_size = (128, 32)
    max_text_len = 32

    def __init__(self, char_list, decoder_type='best_path', must_restore=False):
        super(Model, self).__init__()
        self.char_list = char_list
        self.decoder_type = decoder_type
        self.must_restore = must_restore
        self.snap_id = 0

        # setup CNN, RNN and CTC
        self.cnn = self.setup_cnn()
        self.rnn = self.setup_rnn()
        self.fc = self.setup_fc()

        # setup optimizer to train NN
        self.criterion = nn.CTCLoss(blank=len(self.char_list))
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.001)

    @staticmethod
    def setup_cnn():
        """
        create CNN layers and return them
        :return:
        """
        cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),
        )
        return cnn

    @staticmethod
    def setup_rnn():
        """create RNN layers and return them"""
        rnn = nn.LSTM(256, 256, num_layers=2, bidirectional=True, batch_first=True)
        return rnn

    def setup_fc(self):
        """create fully connected layer and return it"""
        fc = nn.Linear(512, len(self.char_list) + 1)
        return fc

    def forward(self, x):
        """forward pass"""
        x = x.unsqueeze(1)  # add channel dimension
        x = self.cnn(x)
        x = x.permute(0, 3, 1, 2)  # move width to sequence dimension
        x = x.squeeze(2)  # remove height dimension
        x, _ = self.rnn(x)
        x = self.fc(x)
        x = x.permute(1, 0, 2)  # move sequence dimension to first for CTC loss
        return x

    def to_sparse(self, texts):
        """put ground truth texts into sparse tensor for ctc_loss"""
        indices = []
        values = []
        shape = [len(texts), 0]  # last entry must be max(labelList[i])

        # go over all texts
        for (batchElement, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            label_str = [self.char_list.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(label_str) > shape[1]:
                shape[1] = len(label_str)
            # put each label into sparse tensor
            for (i, label) in enumerate(label_str):
                indices.append([batchElement, i])
                values.append(label)

        return indices, values, shape  # for label in ctc loss

    def decoder_output_to_text(self, ctc_output, batch_size):
        """extract texts from output of CTC decoder"""
        encoded_label_strs = [[] for _ in range(batch_size)]
        blank = len(self.char_list)
        for b in range(batch_size):
            for label in ctc_output[b]:
                if label == blank:
                    break
                encoded_label_strs[b].append(label)
        return ["".join([self.charList[c] for c in labelStr]) for labelStr in encoded_label_strs]

    def train_batch(self, batch):
        """feed a batch into the NN to train it"""
        self.train()
        num_batch_elements = len(batch['imgs'])
        sparse = self.toSparse(batch['gtTexts'])
        imgs = torch.tensor(batch['imgs'], dtype=torch.float32)
        gt_texts = (torch.LongTensor(sparse[0]), torch.LongTensor(sparse[1]), torch.Size(sparse[2]))
        seq_len = torch.tensor([Model.maxTextLen] * num_batch_elements, dtype=torch.int32)

        self.optimizer.zero_grad()
        preds = self.forward(imgs)
        loss = self.criterion(preds, gt_texts, seq_len, torch.IntTensor([Model.maxTextLen] * num_batch_elements))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def infer_batch(self, batch, calc_probability=False, probability_of_gt=False):
        """feed a batch into the NN to recognize the texts"""
        self.eval()
        num_batch_elements = len(batch['imgs'])
        imgs = torch.tensor(batch['imgs'], dtype=torch.float32)
        seq_len = torch.tensor([Model.maxTextLen] * num_batch_elements, dtype=torch.int32)

        with torch.no_grad():
            preds = self.forward(imgs)
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)

        texts = self.decoderOutputToText(preds, num_batch_elements)

        probs = None
        if calc_probability:
            sparse = self.toSparse(batch['gtTexts']) if probability_of_gt else self.toSparse(texts)
            gt_texts = (torch.LongTensor(sparse[0]), torch.LongTensor(sparse[1]), torch.Size(sparse[2]))
            loss = self.criterion(preds, gt_texts, seq_len, torch.IntTensor([Model.maxTextLen] * num_batch_elements))
            probs = torch.exp(-loss).cpu().numpy()

        return texts, probs

    def save(self):
        """save model to file"""
        self.snap_id += 1
        torch.save(self.state_dict(), f'../model/snapshot_{self.snap_id}.pth')

    def load(self, path):
        """load model from file"""
        self.load_state_dict(torch.load(path))
