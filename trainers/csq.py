from trainers.dpn import DPNTrainer
from trainers.orthohash import OrthoHashTrainer


class CSQTrainer(DPNTrainer):
    """
    CSQ is very similar to DPN, so we can just use DPNTrainer as parent

    """

    def load_model(self):
        super(OrthoHashTrainer, self).load_model()
