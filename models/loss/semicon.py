from models.loss.base import BaseLoss


class ADSHLoss(BaseLoss):

    def __init__(self, nbit, gamma=200, **kwargs):
        super(ADSHLoss, self).__init__()
        self.nbit = nbit
        self.gamma = gamma

    def forward(self, F, B, S, omega):
        """

        :param F: train_codes
        :param B: retrieval_codes
        :param S: similarity matrix
        :param omega: sample index
        :return:
        """
        hash_loss = ((self.nbit * S - F @ B.t()) ** 2).sum() / (F.shape[0] * B.shape[0]) / self.nbit * 12
        quantization_loss = ((F - B[omega, :]) ** 2).sum() / (F.shape[0] * B.shape[0]) * self.gamma / self.nbit * 12

        self.losses['hash'] = hash_loss
        self.losses['quan'] = quantization_loss

        loss = hash_loss + quantization_loss
        return loss
