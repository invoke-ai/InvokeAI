---
Class: ai
Topic: Invoke Doc
Subject: Installation
---


Installing InvokeAi on a MAC
# Requirements (MAC)


| RAM   | DISK   | CPU                   |
| ----- | ------ | --------------------- |
| 12 GB | 1 TB   | M1 or M2              |
|       | 103G + | for models and Invoke |
|       |        |                       |

## 1. Install Xcode

The easiest method to download Xcode is to open up the _App Store_ application on your desktop, search for _“Xcode”_ in the search bar, and then click the _“Get”_ button.

After installing Xcode you’ll want to open up a terminal and ensure you have [accepted the developer license](http://apple.stackexchange.com/questions/175069/how-to-accept-xcode-license) and install the command line developer tools.

This needs to be done so that opencv will compile when it is called by InvokeAI.


## 2. Install HomeBrew for Apple Silicone

If you use Homebrew on your MAC to install applications, you will need the version of HomeBrew made for Apple Silicone rather than the original Intel version. While you can install Patchmatch with the Intel Version, it did not work in Invoke and Invoke would not be able to do inpainting or out-painting. 

To find out which version you have:

```
brew config

```

If you have an x86 installation of Homebrew, "HOMEBREW_PREFIX" will point to "usr/local"

If you have the arm64 installation of HomeBrew, "HOMEBREW_PREFIX" will point to "opt/homebrew"

If you need to remove the x89 version 

```
 `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/uninstall.sh)"`
    
```

To check if the uninstall was successful

```
    `brew --version`

```

It should say "command not found" if the above command was successful at removing it.



## 3. Install opencv using brew

```
arch -arm64 brew install opencv

```


## 4. Install Python 3.11 using brew


Invoke requires python 3.10 or 3.11. If you don't already have one of these versions installed, we suggest installing 3.11, as it will be supported for longer.

```
brew install python@3.11

```

To see where python was installed:

```
ls -F1 /opt/homebrew/bin/p*

```


## 5. Download the Invoke Automated Installation Script

[[Getting the Latest Installer_draft_v1]] and running it for the first time.





