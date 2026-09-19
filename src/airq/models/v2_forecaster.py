import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=kernel_size,
                              padding=padding,
                              bias=bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next

class V2Forecaster(nn.Module):
    def __init__(self, s5p_channels=3, s2_channels=12, hidden_dim=64):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Encoder uses S5P + S2
        self.encoder_cell = ConvLSTMCell(input_dim=s5p_channels + s2_channels, 
                                         hidden_dim=hidden_dim)
        
        # Decoder uses previous S5P prediction
        self.decoder_cell = ConvLSTMCell(input_dim=s5p_channels, 
                                         hidden_dim=hidden_dim)
                                         
        self.output_conv = nn.Conv2d(hidden_dim, s5p_channels, kernel_size=1)
        
    def forward(self, p1_x, s2_x, k=30):
        # p1_x: (B, 12, 3, H, W)
        # s2_x: (B, 12, 12, H, W)
        
        B, seq_len, _, H, W = p1_x.shape
        
        # Initialize hidden state
        h_t = torch.zeros(B, self.hidden_dim, H, W, device=p1_x.device)
        c_t = torch.zeros(B, self.hidden_dim, H, W, device=p1_x.device)
        
        # Encode History
        for t in range(seq_len):
            x_t = torch.cat([p1_x[:, t], s2_x[:, t]], dim=1)
            h_t, c_t = self.encoder_cell(x_t, (h_t, c_t))
            
        # Decode Future
        outputs = []
        decoder_input = p1_x[:, -1] # Start with last known S5P
        
        for t in range(k):
            h_t, c_t = self.decoder_cell(decoder_input, (h_t, c_t))
            pred = self.output_conv(h_t)
            outputs.append(pred)
            decoder_input = pred # Auto-regressive
            
        return torch.stack(outputs, dim=1) # (B, K, 3, H, W)

