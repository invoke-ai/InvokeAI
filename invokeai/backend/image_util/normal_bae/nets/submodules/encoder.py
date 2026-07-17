import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

import invokeai.backend.util.logging as logger

# Emit the TorchScript-deprecation breadcrumb only once per process, rather than on every model load.
_warned_jit_deprecation = False


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        basemodel_name = 'tf_efficientnet_b5_ap'
        print('Loading base model ()...'.format(basemodel_name), end='')
        repo_path = os.path.join(os.path.dirname(__file__), 'efficientnet_repo')
        # The vendored EfficientNet (geffnet) code uses @torch.jit.script / @torch.jit.script_method, which
        # are deprecated in Python 3.14+ and emit a warning for every decorated function on each load (~26
        # per model). We can't rewrite this third-party TorchScript code to torch.compile/torch.export, so
        # suppress the per-call spam but surface a single breadcrumb once per process so it stays on our
        # radar. When torch eventually *removes* these APIs, torch.hub.load will raise rather than warn, so
        # the Normal Map preprocessor will fail loudly at that point regardless of this suppression.
        global _warned_jit_deprecation
        if not _warned_jit_deprecation:
            logger.warning(
                "Normal Map (normal_bae) loads a vendored EfficientNet that uses torch.jit.script, which is "
                "deprecated in Python 3.14+ and will eventually be removed from torch. It still works for "
                "now; this needs a migration before torch drops the TorchScript APIs."
            )
            _warned_jit_deprecation = True
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*torch.jit.script.*", category=DeprecationWarning)
            basemodel = torch.hub.load(repo_path, basemodel_name, pretrained=False, source='local')
        print('Done.')

        # Remove last layer
        print('Removing last two layers (global_pool & classifier).')
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        self.original_model = basemodel

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if (k == 'blocks'):
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


