from PatchTST import Model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Config:
    def __init__(self, task_name, seq_len, pred_len, d_model, dropout, factor,
                 output_attention, n_heads, d_ff, activation, e_layers):
        self.task_name = task_name
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.dropout = dropout
        self.factor = factor
        self.output_attention = output_attention
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.activation = activation
        self.e_layers = e_layers

# Creating an instance of Config
configs = Config(
    task_name='anomaly_detection',
    seq_len=1000,
    pred_len=256,
    d_model=None,
    dropout=0.1,
    factor=0.001,
    output_attention=None,
    n_heads=12,
    d_ff=None,
    activation=None,
    e_layers=None
)

model = Model(configs)

print(model)