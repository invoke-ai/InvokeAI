"""
Adapted from https://github.com/XPixelGroup/BasicSR
License: Apache-2.0

As of Feb 2024, `basicsr` appears to be unmaintained. It imports a function from `torchvision` that is removed in
`torchvision` 0.17. Here is the deprecation warning:

    UserWarning: The torchvision.transforms.functional_tensor module is deprecated in 0.15 and will be **removed in
    0.17**. Please don't rely on it. You probably just need to use APIs in torchvision.transforms.functional or in
    torchvision.transforms.v2.functional.

As a result, a dependency on `basicsr` means we cannot keep our `torchvision` dependency up to date.

Because we only rely on a single class `RRDBNet` from `basicsr`, we've copied the relevant code here and removed the
dependency on `basicsr`.

The code is almost unchanged, only a few type annotations have been added. The license is also copied.
"""
