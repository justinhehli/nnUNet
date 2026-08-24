import torch
from torch import nn
from torch.nn import functional as F


class AdaptiveWingLoss(nn.Module):
    """
    Adaptive Wing loss for heatmap regression (https://ieeexplore.ieee.org/document/9010657)

    Default hyperparameters (omega=14, theta=0.5, epsilon=1, alpha=2.1) are the paper's own choice
    (Sec. 4.2, confirmed by ablation in Sec. 7.6.1 / Tables 5-6).
    """

    def __init__(self, omega: float = 14.0, theta: float = 0.5, epsilon: float = 1.0,
                 alpha: float = 2.1, nonlinearity=torch.sigmoid):
        super().__init__()
        assert alpha > 2, "alpha must be slightly larger than 2 for y in [0, 1]"
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.nonlinearity = nonlinearity

    def _per_pixel_loss(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        delta = (y - y_hat).abs()
        exponent = self.alpha - y
        theta_over_eps = self.theta / self.epsilon

        nonlinear = self.omega * \
            torch.log1p((delta / self.epsilon) ** exponent)

        A = self.omega * (1.0 / (1.0 + theta_over_eps ** exponent)) * exponent \
            * (theta_over_eps ** (exponent - 1.0)) * (1.0 / self.epsilon)
        C = self.theta * A - self.omega * \
            torch.log1p(theta_over_eps ** exponent)
        linear = A * delta - C

        return torch.where(delta < self.theta, nonlinear, linear)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_hat = self.nonlinearity(
            net_output) if self.nonlinearity is not None else net_output
        return self._per_pixel_loss(target, y_hat).mean()


class WeightedAdaptiveWingLoss(nn.Module):
    """
    AdaptiveWingLoss wrapped with the paper's Weighted Loss Map (Sec. 4.3, Eq. 4-5), which upweights
    foreground pixels and "difficult" background pixels (background pixels close to foreground) so
    the (heavily background-dominated) heatmap doesn't drown out the pixels that actually matter for
    localization:

        M = 1 where dilate_3x3(y) >= dilation_threshold else 0
        loss = AWing(y, yhat) * (weight * M + 1)

    "gray dilation" (paper's term for grayscale/continuous-valued morphological dilation) is a local
    3x3 max-filter, implemented here via max_pool2d. Default weight=10, dilation_threshold=0.2 are
    the paper's own (Sec. 4.3). In the paper's own ablation (Table 7), this improves on AWing alone
    by a further 0.35% NME (4.65% -> 4.30%), which is why it's the default here too.
    """

    def __init__(self, omega: float = 14.0, theta: float = 0.5, epsilon: float = 1.0,
                 alpha: float = 2.1, nonlinearity=torch.sigmoid,
                 weight: float = 10.0, dilation_threshold: float = 0.2):
        super().__init__()
        self.awing = AdaptiveWingLoss(omega=omega, theta=theta, epsilon=epsilon, alpha=alpha,
                                      nonlinearity=nonlinearity)
        self.weight = weight
        self.dilation_threshold = dilation_threshold

    def forward(self, net_output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_hat = self.awing.nonlinearity(
            net_output) if self.awing.nonlinearity is not None else net_output
        per_pixel = self.awing._per_pixel_loss(target, y_hat)

        dilated = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
        loss_mask = (dilated >= self.dilation_threshold).to(per_pixel.dtype)
        weight_map = self.weight * loss_mask + 1.0

        return (per_pixel * weight_map).mean()


class ReweightedAdaptiveWingLoss(nn.Module):
    """
    AdaptiveWingLoss with two independent, optional per-pixel reweighting schemes meant to address
    confident false negatives (raw output near 0 at true landmark pixels), on top of the paper's
    Weighted Loss Map (WeightedAdaptiveWingLoss) if use_weight_map is set:

      - focal-style hard-positive upweighting (gamma is not None): at GT-positive pixels, the
        per-pixel loss is multiplied by (1 - pred)^gamma, pred = sigmoid(net_output) detached before
        computing the modulating factor (so the network can't reduce its own loss weight just by
        pushing pred down without shrinking delta). Standard RetinaNet-style focal reweighting,
        applied per-pixel instead of per-box.

      - soft-sampling background down-weighting (max_downweight is not None): at GT-background
        pixels, the per-pixel loss is scaled by (1 - snapshot_confidence), where snapshot_confidence
        = sigmoid(snapshot_output) comes from a frozen, periodically-refreshed snapshot of the model
        (see nnUNetTrainerHeatmapAdaptiveWingSoftSampling - refreshed every N epochs rather than every
        step, to avoid a feedback loop where the model's own current, possibly-wrong confidence erases
        its own training signal). Down-weighting is floored at (1 - max_downweight) so a background
        pixel can never be fully zeroed out of the loss. snapshot_output must be passed to forward()
        whenever max_downweight is set - there is no silent no-reweighting fallback, since that would
        just look like soft-sampling doing nothing.

    Both schemes can be enabled together (their weights multiply).
    """

    def __init__(self, omega: float = 14.0, theta: float = 0.5, epsilon: float = 1.0,
                 alpha: float = 2.1, nonlinearity=torch.sigmoid,
                 gamma: float = None, max_downweight: float = None,
                 use_weight_map: bool = True, weight: float = 10.0, dilation_threshold: float = 0.2):
        super().__init__()
        assert gamma is None or gamma >= 0, "gamma must be >= 0"
        assert max_downweight is None or 0.0 <= max_downweight <= 1.0, \
            "max_downweight must be in [0, 1]"
        self.awing = AdaptiveWingLoss(omega=omega, theta=theta, epsilon=epsilon, alpha=alpha,
                                      nonlinearity=nonlinearity)
        self.gamma = gamma
        self.max_downweight = max_downweight
        self.use_weight_map = use_weight_map
        self.weight = weight
        self.dilation_threshold = dilation_threshold

    def forward(self, net_output: torch.Tensor, target: torch.Tensor,
                snapshot_output: torch.Tensor = None) -> torch.Tensor:
        nonlin = self.awing.nonlinearity
        y_hat = nonlin(net_output) if nonlin is not None else net_output
        per_pixel = self.awing._per_pixel_loss(target, y_hat)

        positive_mask = (target > 0).to(per_pixel.dtype)
        weight_map = torch.ones_like(per_pixel)

        if self.gamma is not None:
            modulating_factor = (1.0 - y_hat.detach()) ** self.gamma
            weight_map = weight_map * \
                (positive_mask * modulating_factor + (1.0 - positive_mask))

        if self.max_downweight is not None:
            assert snapshot_output is not None, \
                "max_downweight is set but no snapshot_output was passed to forward() - wire the " \
                "frozen snapshot's prediction through (see nnUNetTrainerHeatmapAdaptiveWingSoftSampling)"
            snapshot_confidence = nonlin(
                snapshot_output).detach() if nonlin is not None else snapshot_output.detach()
            downweight = snapshot_confidence.clamp(max=self.max_downweight)
            background_mask = 1.0 - positive_mask
            weight_map = weight_map * \
                (background_mask * (1.0 - downweight) + positive_mask)

        if self.use_weight_map:
            dilated = F.max_pool2d(target, kernel_size=3, stride=1, padding=1)
            loss_mask = (dilated >= self.dilation_threshold).to(per_pixel.dtype)
            weight_map = weight_map * (self.weight * loss_mask + 1.0)

        return (per_pixel * weight_map).mean()
