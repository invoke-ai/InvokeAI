---
Class: ai
Topic: InvokeAI Official Doc
Document Section: Installation
Created: 2024-07-08
Published to My Github: true
Pull Request: 
Author: Smile4yourself
---

## Updating

Updating is exactly the same as installing - download the latest installer, run the script, and select the latest version.

Detailed procedure is here [Getting the Latest Installer for InvokeAI](/Install/Getting_the_Latest_Installer_for_InvokeAI)

Using this way to install may avoid dependancy problems when upgrading packages.

Dependency Resolution Issues

We've found that pip's dependency resolution can cause issues when upgrading packages. One very common problem was pip "downgrading" torch from CUDA to CPU, but things broke in other novel ways.

The installer script doesn't have this kind of problem, so we are using it for updating InvokeAI as well as installing it the first time.

[FAQ -Installation Issues](/Install/FAQ_Installation_Issues)

