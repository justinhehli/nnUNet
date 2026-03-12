
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftSkeletonize(torch.nn.Module):
    def __init__(self, num_iter: int):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
        p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
        return torch.min(p1, p2)

    def soft_dilate(self, img):
        return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))

    def soft_open(self, img):
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        img1 = self.soft_open(img)
        skel = F.relu(img - img1)

        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)

        return skel

    def forward(self, img):
        # input img shape: (B, C, H, W), where C = 1 for binary masks (foreground only)
        return self.soft_skel(img)


class SoftClDiceLoss(nn.Module):
    def __init__(self, skeletonize_iter: int, smooth: float = 1e-6):
        super(SoftClDiceLoss, self).__init__()
        self.iter = skeletonize_iter
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=skeletonize_iter)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        mask_prob = torch.sigmoid(logits)

        skel_pred = self.soft_skeletonize(mask_prob)
        skel_true = self.soft_skeletonize(targets)

        tprec = (torch.sum(skel_pred * targets) +
                 self.smooth) / (torch.sum(skel_pred) + self.smooth)
        tsens = (torch.sum(skel_true * mask_prob) +
                 self.smooth) / (torch.sum(skel_true) + self.smooth)

        return 1. - 2.0 * (tprec * tsens) / (tprec + tsens)
