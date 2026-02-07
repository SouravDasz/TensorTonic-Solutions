import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Write code here
    x_t=np.array(x_t)
    h_prev=np.array(h_prev)
    b=np.array(b)
    inp=x_t@Wx
    hid=h_prev@Wh
    z=inp+hid+b
    out=np.tanh(z)
    return out
