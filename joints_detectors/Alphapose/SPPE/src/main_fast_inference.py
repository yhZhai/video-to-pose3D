import sys

import torch
import torch._utils
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed

from SPPE.src.models.FastPose import createModel
from SPPE.src.utils.img import flip, shuffleLR

try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


class InferenNet(nn.Module):
    def __init__(self, kernel_size, dataset):
        super(InferenNet, self).__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = createModel().to(device)
        print('Loading pose model from {}'.format('joints_detectors/Alphapose/models/sppe/duc_se.pth'))
        sys.stdout.flush()
        model.load_state_dict(torch.load('joints_detectors/Alphapose/models/sppe/duc_se.pth', 
                                         map_location="cpu"))
        model.eval()
        self.pyranet = model

        self.dataset = dataset

    def forward(self, x):
        out = self.pyranet(x)
        out = out.narrow(1, 0, 17)

        flip_out = self.pyranet(flip(x))
        flip_out = flip_out.narrow(1, 0, 17)

        flip_out = flip(shuffleLR(
            flip_out, self.dataset))

        out = (flip_out + out) / 2

        return out


class InferenNet_fast(nn.Module):
    def __init__(self, kernel_size, dataset):
        super(InferenNet_fast, self).__init__()

        model = createModel().cuda()
        print('Loading pose model from {}'.format('models/sppe/duc_se.pth'))
        model.load_state_dict(torch.load('models/sppe/duc_se.pth'))
        model.eval()
        self.pyranet = model

        self.dataset = dataset

    def forward(self, x):
        out = self.pyranet(x)
        out = out.narrow(1, 0, 17)

        return out
