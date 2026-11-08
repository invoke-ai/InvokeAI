"""Fun-style control branches (alibaba-pai VideoX-Fun), run beside a base transformer through forward hooks.

Every Fun control model follows one recipe: the control context is projected to the block width, runs through a short
chain of blocks that are the base transformer's block plus two projections, and each block's ``after_proj`` output is
added to the base image stream after a listed base layer. ``branch.py`` holds the recipe, the stages a control runs in
and the seams a single-chain family fills in; ``hooks.py`` attaches a control to a transformer without touching its
forward; a module per family (``qwen_image.py``, ``flux2.py``, ``z_image.py``) supplies the block class, the layer map
and the calling convention. ``conditioning.py`` builds the control context all of them were trained on.
"""
