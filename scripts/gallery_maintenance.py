#!/usr/bin/env python3
"""
gallery_maintenance.py

Remove orphan images from the gallery directory.
Remove orphan database entries for images that no longer exist in the gallery directory.
Regenerate missing thumbnail images.
"""

from invokeai.backend.util.gallery_maintenance import main

main()
