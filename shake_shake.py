import torch

class ShakeShake(torch.autograd.Function):
    def __init__(self):
        super().__init__()

    @staticmethod
    def _get_random_alpha(x):
        alpha = 0.5+0.1*(2*torch.cuda.FloatTensor(x.size(0)).uniform_()-1)
        alpha = alpha.view(alpha.size(0),1,1,1).expand_as(x)
        return alpha

    @staticmethod
    def forward(ctx,x,y, training=True):
        if training:
            alpha=ShakeShake._get_random_alpha(x)
        else:
            alpha=0.5
        return alpha*x+(1-alpha)*y


    @staticmethod
    def backward(ctx,grad_by_forward_output):

        alpha = ShakeShake._get_random_alpha(grad_by_forward_output)

        d_forward_output_by_x=alpha
        d_forward_output_by_y = 1-alpha
        grad_by_x=grad_by_forward_output*d_forward_output_by_x
        grad_by_y=grad_by_forward_output*d_forward_output_by_y
        return grad_by_x,grad_by_y, None

