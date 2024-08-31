class DecoderType:
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2

class RNNModel(nn.Module):
    "PyTorch model for HTR - Support Devanagari Language"

    # model constants
    batchSize = 25
    imgSize = (128, 32)
    maxTextLen = 32

    def __init__(self, charList, decoderType=DecoderType.BestPath, mustRestore=False):
        super().__init__()
        self.charList = charList
        self.decoderType = decoderType
        self.mustRestore = mustRestore

        # CNN layers
        self.cnn = nn.Sequential(
            self._conv_layer(1, 32),
            self._conv_layer(32, 64),
            self._conv_layer(64, 128),
            self._conv_layer(128, 128),
            self._conv_layer(128, 256),
        )

        # RNN layers
        self.rnn = nn.LSTM(256, 256, num_layers=2, bidirectional=True)

        # Fully connected layer
        self.fc = nn.Linear(512, len(self.charList))  # +1 for CTC blank
        
        # Criterion and Optimizer
        self.criterion = nn.CTCLoss(blank = 0)
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=0.001)

    def _conv_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        # CNN
        x = self.cnn(x)  # (B, C, H, W)
        
        # Prepare for RNN
        x = x.permute(0, 2, 1, 3)
        #x = x.permute(3, 0, 1, 2)  # (W, B, C, H)
        x = x.flatten(2)  # (W, B, C*H)
        print(x.shape)

        # RNN
        x, state = self.rnn(x)  # (W, B, 2*H)

        # FC
        x = self.fc(x)  # (W, B, num_classes)

        # Log softmax
        x = F.log_softmax(x, dim=2)

        return x

    def to_sparse(self, texts):
        """Convert ground truth texts into sparse tensor for ctc_loss"""
        indices = []
        values = []
        shape = [len(texts), 0]  # last entry must be max(labelList[i])

        # Go over all texts
        for batchElement, text in enumerate(texts):
            # Convert to string of label (i.e. class-ids)
            labelStr = [self.charList.index(c) for c in text]  # +1 because 0 is reserved for blank
            # Sparse tensor must have size of max. label-string
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            # Put each label into sparse tensor
            for i, label in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)
        sparse = (indices, values, shape)

        return torch.sparse_coo_tensor(torch.LongTensor(sparse[0]).t(), 
                                       torch.LongTensor(sparse[1]), 
                                       torch.Size(sparse[2])).to_dense()

    def decoder_output_to_text(self, ctc_output, batch_size):
        """Extract texts from output of CTC decoder"""
        
        # Contains string of labels for each batch element
        encoded_label_strs = [[] for _ in range(batch_size)]

        # CTC returns tuple, first element is FloatTensor
        decoded = ctc_output[0]

        # Go over all indices and save mapping: batch -> values
        for (idx, idx2d) in enumerate(decoded.indices):
            label = decoded.values[idx].item()
            batch_element = idx2d[0].item()  # index according to [b,t]
            encoded_label_strs[batch_element].append(label)

        # Map labels to chars for all batch elements
        return [''.join([self.charList[c - 1] for c in labelStr if c != 0]) for labelStr in encoded_label_strs]

    def train_batch(self, batch):
        self.train()
        self.optimizer.zero_grad()
        
        images, texts = batch.imgs, batch.gt_texts
        images = torch.tensor(images).unsqueeze(1)
        images = torch.tensor(images, dtype = torch.float32)
        log_probs = self(images)
        print(f"The log probabilities are of shape {log_probs.shape}")
        gt_texts = self.to_sparse(texts)
        input_lengths = torch.full(size=(log_probs.size(1),), fill_value=log_probs.size(0), dtype=torch.long)
        target_lengths = torch.LongTensor([len(t) for t in batch.gt_texts])
        loss = self.criterion(log_probs, targets, input_lengths, target_lengths)
        
        loss.backward()
        optimizer.step()
        
        return loss.item()

    def infer_batch(self, batch):
        self.eval()
        with torch.no_grad():
            imgs = batch
            log_probs = self(imgs)
            
        if self.decoderType == DecoderType.BestPath:
            decoded = log_probs.argmax(dim=2).permute(1, 0)
            texts = self.decoder_output_to_text(decoded, batch.size(0))
        elif self.decoderType == DecoderType.BeamSearch:
            decoded = torch.nn.functional.ctc_beam_search_decoder(log_probs.permute(1, 0, 2), beam_width=100)
            texts = self.decoder_output_to_text(decoded, batch.size(0))
        else:
            raise NotImplementedError("WordBeamSearch not implemented in PyTorch version")
        
        return texts

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

# Initialize an instance of the model
model = RNNModel(dataloader.char_list)
# Test the model
dummy_input = torch.randn(32, 1, 128, 32)
output = model(dummy_input)
print("Model output shape:", output.shape)
summary(model, (1, 128, 32))

# Forward pass of the CNN
print(f"The images are of shape {images.shape}")
cnn_output = model.cnn(images)
print("\n")
print(f"After the CNN, the output is of shape {cnn_output.shape}")
# Permute and Flatten operations for the RNN
cnn_output = cnn_output.permute(0, 2, 1, 3)
print("\n")
print(f"After rearranging the tensor, the shape is {cnn_output.shape}")
cnn_output = cnn_output.flatten(2)
print(f"After flattening the last two layers, the shape is {cnn_output.shape}")
print("\n")
print(f"Applying RNN")
rnn_output, _ = nn.LSTM(256, 256, num_layers = 2, bidirectional = True)(cnn_output)
print(f"After applying the LSTM in the RNN the shape is {rnn_output.shape}")
print("\n")
logits = model.fc(rnn_output)
print(f"After applying a linear layer the output is {logits.shape}")
print("\n")
print("Applying Log Softmax")
log_probs = F.log_softmax(logits, dim = -1)
print(f"Afer applying log softmax the shape is {logits.shape}")