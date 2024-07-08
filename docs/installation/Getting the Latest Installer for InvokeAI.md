---
Class: ai
Topic: InvokeAI Official Doc
Document Section: Installation
Created: 2024-07-08
Published to My Github: true
Pull Request: 
Author: Smile4yourself
---
%%
linked from three different files: (Installation Windows, Installation Linux, Installation MAC)

%%

This Doc is part of the Installation Document for InvokeAI, and applies to Windows, Linux, and MAC

## Getting the Latest Installer

Download the `InvokeAI-installer-vX.Y.Z.zip` file from the [latest release](https://github.com/invoke-ai/InvokeAI/releases/latest) page. It is at the bottom of the page, under **Assets**.

Alternatively here are the latest:

[InvokeAI-installer-v4.2.4.zip](https://github.com/invoke-ai/InvokeAI/releases/download/v4.2.4/InvokeAI-installer-v4.2.4.zip)

After unzipping the installer, you should have a `InvokeAI-Installer` folder with some files inside, including `install.bat` and `install.sh`.

## Running the Installer


Windows users should first double-click the `WinLongPathsEnabled.reg` file to prevent a failed installation due to long file paths.

Double-click the "install.bat" if you are a windows user, or if you are using Linux or a MAC open the terminal and type:

```
./install.sh

```

You may get a popup saying the file comes from an `Untrusted Publisher`. Click `More Info` and `Run Anyway` to get past this.

The installation script is simple, with only a few prompts:

-   Select the version to install. Unless you have a specific reason to install a specific version, select the default (the latest version).
-   Select location for the install. 
-  If you use the same folder has the previous InvokeAI version, the new version will load all your images and models.
-  Be sure you have enough space if it is the first time you run it. 
-  Select a GPU device if you have one.

Slow Installation

The installer needs to download several GB of data and install it all. It may appear to get stuck at 99.9% when installing `pytorch` or during a step labeled "Installing collected packages".

If it is stuck for over 10 minutes, something has probably gone wrong and you should close the window and restart.

## Running the Application

Find the install location you selected earlier. Double-click the `invoke.bat` to run the app if in Windows, or if in Linux or on a MAC, open the terminal and type:

```
./invoke.sh

```

Choose the first option to run the UI. After a series of startup messages, you'll see something like this:

```
Uvicorn running on http://127.0.0.1:9090 (Press CTRL+C to quit)
```

Copy the URL into your browser and you should see the User Interface (UI) and now you can begin using the graphical interface to InvokeAI.


## First-time Setup

🙏This is a good place to point to a video that introduces the UI and how to download models.

You will need to [install some models](https://invoke-ai.github.io/InvokeAI/installation/050_INSTALLING_MODELS/) before you can generate. 

Check the [configuration docs](https://invoke-ai.github.io/InvokeAI/features/CONFIGURATION/) for details on configuring the application.


[[How to Update InvokeAI]]

[[FAQ -Installation Issues]]




