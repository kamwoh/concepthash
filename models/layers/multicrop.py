import torch
import torch.nn as nn


class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.

    copy from https://github.com/facebookresearch/dino/blob/main/utils.py
    """

    def __init__(self, backbone, hash_fc, head):
        super(MultiCropWrapper, self).__init__()

        self.backbone = backbone
        self.hash_fc = hash_fc
        self.head = head

    def forward(self, x):
        """

        Return:
            representations, codes, projs
        """
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, codes = 0, torch.empty(0).to(x[0].device)
        representations = torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _repr = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_repr, tuple):
                _repr = _repr[0]

            _codes = self.hash_fc(_repr)

            # accumulate outputs
            codes = torch.cat((codes, _codes))
            representations = torch.cat((representations, _repr))
            start_idx = end_idx

        # Run the head forward on the concatenated features.
        projs = self.head(codes)
        return representations, codes, projs
