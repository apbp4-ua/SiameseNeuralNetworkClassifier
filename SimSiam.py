from torch import nn
import timm
import torch.nn.functional as F


class SimSiam(nn.Module):
  def __init__(self, proj_dim=2048, pred_dim=512):
    super().__init__()
    self.backbone = timm.create_model("convnext_base", pretrained=True)
    self.backbone.reset_classifier(0)
    self.projector = nn.Sequential(
        nn.Linear(proj_dim, proj_dim),
        nn.BatchNorm1d(proj_dim),
        nn.ReLU(),
        nn.Linear(proj_dim, proj_dim)
    )

    self.predictor = nn.Sequential(nn.Linear(proj_dim, pred_dim),
        nn.BatchNorm1d(pred_dim),
        nn.ReLU(),
        nn.Linear(proj_dim, pred_dim))

  def forward(self, x1, x2):
    h1 = self.backbone(x1)
    h2 = self.backbone(x2)
    z1 = self.projector(h1)
    z2 = self.projector(h2)
    p1 = self.predictor(z1)
    p2 = self.predictor(z2)

    return z1, z2, p1, p2

  def simsiam_loss(p, z):
      z = z.detach()
      return -F.cosine.similarity(p, z, dim=-1).mean()