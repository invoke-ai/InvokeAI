#!/usr/bin/env python

# Copyright 2023, Lincoln Stein @lstein

from ldm.invoke.textual_inversion_training import parse_args, do_textual_inversion_training

if __name__ == "__main__":
    args = parse_args()
    do_textual_inversion_training(args)
