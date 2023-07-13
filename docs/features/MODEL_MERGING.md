---
title: Model Merging
---

# :material-image-off: Model Merging

## How to Merge Models

As of version 2.3, InvokeAI comes with a script that allows you to
merge two or three diffusers-type models into a new merged model. The
resulting model will combine characteristics of the original, and can
be used to teach an old model new tricks.

You may run the merge script by starting the invoke launcher
(`invoke.sh` or `invoke.bat`) and choosing the option for _merge
models_. This will launch a text-based interactive user interface that
prompts you to select the models to merge, how to merge them, and the
merged model name.

Alternatively you may activate InvokeAI's virtual environment from the
command line, and call the script via `merge_models --gui` to open up
a version that has a nice graphical front end. To get the commandline-
only version, omit `--gui`.

The user interface for the text-based interactive script is
straightforward. It shows you a series of setting fields. Use control-N (^N)
to move to the next field, and control-P (^P) to move to the previous
one. You can also use TAB and shift-TAB to move forward and
backward. Once you are in a multiple choice field, use the up and down
cursor arrows to move to your desired selection, and press <SPACE> or
<ENTER> to select it. Change text fields by typing in them, and adjust
scrollbars using the left and right arrow keys.

Once you are happy with your settings, press the OK button. Note that
there may be two pages of settings, depending on the height of your
screen, and the OK button may be on the second page. Advance past the
last field of the first page to get to the second page, and reverse
this to get back.

If the merge runs successfully, it will create a new diffusers model
under the selected name and register it with InvokeAI.

## The Settings

* Model Selection -- there are three multiple choice fields that
  display all the diffusers-style models that InvokeAI knows about.
  If you do not see the model you are looking for, then it is probably
  a legacy checkpoint model and needs to be converted using the
  `invoke` command-line client and its `!optimize` command. You
  must select at least two models to merge. The third can be left at
  "None" if you desire.

* Alpha -- This is the ratio to use when combining models. It ranges
  from 0 to 1. The higher the value, the more weight is given to the
  2d and (optionally) 3d models. So if you have two models named "A"
  and "B", an alpha value of 0.25 will give you a merged model that is
  25% A and 75% B.

* Interpolation Method -- This is the method used to combine
  weights. The options are "weighted_sum" (the default), "sigmoid",
  "inv_sigmoid" and "add_difference". Each produces slightly different
  results. When three models are in use, only "add_difference" is
  available. (TODO: cite a reference that describes what these
  interpolation methods actually do and how to decide among them).

* Force -- Not all models are compatible with each other. The merge
  script will check for compatibility and refuse to merge ones that
  are incompatible. Set this checkbox to try merging anyway.

* Name for merged model - This is the name for the new model. Please
  use InvokeAI conventions - only alphanumeric letters and the
  characters ".+-".

