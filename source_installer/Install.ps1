#!/usr/bin/env pwsh
$invokeAiVersion = "2.2.3" # this can be dynamic if removed from filename
$installBundleFilename = "invokeAI-src-installer-${invokeAiVersion}-windows.zip"

# TODO: we should prompt the user for the path here
$installFolder = "${home}/invokeai/installer"

$installBundleUrl = "https://github.com/invoke-ai/InvokeAI/releases/latest/download/${installBundleFilename}"

New-Item -itemtype Directory -Path $installFolder -Force
cd $installFolder

# download the install bundle
# below is needed because by default the client uses TLS<1.2, which fails with SChannel on modern websites.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest -Uri $installBundleUrl -UseBasicParsing -Outfile $installBundleFilename;

## Alt way of downloading, untested
# $client = New-Object System.Net.WebClient
# $client.DownloadFile($installBundleUrl, $installBundleFilename)

# extract zip
Expand-Archive -LiteralPath $installBundleFilename -DestinationPath $installFolder -Force

# do the installation
# hardcoded folder name in the zip
cd invokeAI
$process = Start-Process -FilePath "install.bat" -Wait -PassThru
if ($process.ExitCode -ne 0) {
    throw "Installation failed."
}
