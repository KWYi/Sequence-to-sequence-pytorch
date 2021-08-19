# Sequence-to-sequence

The codes of Sequence to seqence.

This code has been used for the paper "Forecast of Major Solar X-Ray Flare Flux Profiles Using Novel Deep Learning Models" (https://doi.org/10.3847/2041-8213/ab701b).

## Usage in the paper
In the paper, following two usages were used.

```
import os, sys
import torch
from seq2seq_models import Encoder4Attn, AttnDecoder_ver3
    
Model_types = ['E_attn_D_ver3']
input_size = 1
unit_sizes = [256]
encoded_sizes = [1]
bidirectional = False
initialize = True
input_len = 30
output_len = 30

E = Encoder4Attn(input_size=input_size, hidden_size=unit_size, layer_num=2,
                 encoded_size=encoded_size, bidirection=bidirectional,
                 drop_frac=drop_ratio, initiallize=initialize).to(device=DEVICE)
                 
D = AttnDecoder_ver3(encoded_size=encoded_size, hidden_size=unit_size,
                     input_len=input_len, layer_num=2, drop_frac=drop_ratio,
                     initiallize=initialize).to(device=DEVICE)
                     
for epoch in EPOCHS:
  for Train_input, Train_target in Train_data_loader:  
    Encoder_result, Encoder_hidden = E(Train_input)
    Decoder_hidden = Encoder_hidden
    Result = torch.zeros(len(Encoder_result), output_len).to(device=DEVICE)
    Result_part = Train_input[:, -1:, :]
    
    for di in range(output_len):  # Save model results in array batch_size*output_len
      # # Output of prior timestep of lstm is used for net input and Encoder result is concated with it in the model
      Result_part, Decoder_hidden, _ = D(Result_part, Decoder_hidden, Encoder_result)
      Result[:, di] = Result_part.view(len(Encoder_result))
```


```
import os, sys
import torch
from seq2seq_models import Encoder, Decoder

Model_types = ['E_D']
input_size = 1
unit_sizes = [192]
encoded_sizes = [8]
bidirectional = False
initialize = True
input_len = 30
output_len = 30

E = Encoder(input_size=input_size, hidden_size=unit_size, layer_num=2,
            encoded_size=encoded_size,
            drop_frac=drop_ratio, initiallize=initialize).to(device=DEVICE)
D = Decoder(input_size=input_size, hidden_size=unit_size, layer_num=2,
            encoded_size=encoded_size,
            drop_frac=drop_ratio, initiallize=initialize).to(device=DEVICE)
                     
for epoch in EPOCHS:
  for Train_input, Train_target in Train_data_loader:  
    Encoder_result, Encoder_hidden = E(Train_input)
    Encoder_result = Encoder_result[:, -1:, :]
    Decoder_hidden = Encoder_hidden
    Result = torch.zeros(len(Encoder_result), output_len).to(device=DEVICE)
    Result_part = Train_input[:, -1:, :]
    
    for di in range(output_len):  # Save model results in array batch_size*output_len
      # # Output of prior timestep of lstm is used for net input and Encoder result is concated with it in the model
      Result_part, Decoder_hidden, _ = D(Result_part, Decoder_hidden, Encoder_result)
      Result[:, di] = Result_part.view(len(Encoder_result))
```
