import torch

def weighted_mse_loss(input, target, weight=1.0):
  """
  计算MSE损失，并对gt中大于0的部分应用额外的权重。

  Args:
    input (torch.Tensor): 预测值。
    target (torch.Tensor): 真实值。
    weight (float): 应用于gt中大于0部分的额外权重。

  Returns:
    torch.Tensor: 加权MSE损失。
  """

  # 计算标准MSE损失
  mse_loss = torch.nn.MSELoss()(input, target)

  # 创建一个掩码，标记gt中大于0的位置
  mask = target > 0

  # 对gt中大于0的部分应用额外权重
  weighted_loss = torch.mean((input[mask] - target[mask]) ** 2) * weight

  # 返回加权MSE损失
  return mse_loss + weighted_loss