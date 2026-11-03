<#
Lecture Live setup. Safe to run again: every step skips what is already done.

  .\setup.ps1             everything, including the models: Whisper (874 MB), speakers (25 MB),
                          Qwen (2.5 GB); downloads resume if interrupted
  .\setup.ps1 -SkipModel  everything except the model downloads

Needs: Python 3.12 (py launcher), Visual Studio 2022 Build Tools with "Desktop development
with C++", CMake, the Vulkan SDK, and the AMD graphics driver.
#>
param([switch]$SkipModel)
$ErrorActionPreference = "Stop"

$Repo = $PSScriptRoot
$Venv = Join-Path $env:USERPROFILE ".venvs\lecture-live"
$Py = Join-Path $Venv "Scripts\python.exe"
$PyW = Join-Path $Venv "Scripts\pythonw.exe"
$BuildRoot = Join-Path $env:USERPROFILE ".venvs\_build"
$PwcVersion = "1.5.1"

function Step([string]$Message) { Write-Host "`n== $Message" -ForegroundColor Cyan }

function Run([string]$Exe, [string[]]$ArgList) {
    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) { throw "$Exe $($ArgList -join ' ') failed with exit code $LASTEXITCODE" }
}

function ConfigValue([string]$Key) {
    $line = Select-String -Path (Join-Path $Repo "config.yaml") -Pattern "^${Key}:\s*`"?([^`"#]*)`"?" | Select-Object -First 1
    if ($null -eq $line) { throw "config.yaml has no '$Key'" }
    return $line.Matches[0].Groups[1].Value.Trim()
}

# ---------------------------------------------------------------------------------------------
Step "Checking prerequisites"
& py -3.12 --version
if ($LASTEXITCODE -ne 0) { throw "Python 3.12 is required (install it from python.org with the py launcher)." }
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) { throw "CMake is not on PATH." }
if (-not $env:VULKAN_SDK -or -not (Test-Path (Join-Path $env:VULKAN_SDK "Bin\glslc.exe"))) {
    throw "The Vulkan SDK is not installed (VULKAN_SDK is not set or has no glslc.exe). Get it from vulkan.lunarg.com."
}
$vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$vc = if (Test-Path $vswhere) { & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath } else { $null }
if (-not $vc) { throw "Visual Studio Build Tools with the C++ workload are required." }
$ErrorActionPreference = "Continue"   # vulkaninfo writes notices to stderr; PowerShell 5.1 would throw on them
$gpu = & vulkaninfo --summary 2>$null | Select-String "deviceName" | Select-Object -First 1
$ErrorActionPreference = "Stop"
if ($null -eq $gpu) { throw "vulkaninfo found no Vulkan GPU. Install or update the AMD graphics driver." }
Write-Host "Vulkan GPU:$($gpu.Line.Split('=')[1])"

# ---------------------------------------------------------------------------------------------
Step "Python environment ($Venv)"
if (-not (Test-Path $Py)) { Run "py" @("-3.12", "-m", "venv", $Venv) }
Run $Py @("-m", "pip", "install", "--upgrade", "--progress-bar", "off", "pip")
Run $Py @("-m", "pip", "install", "--progress-bar", "off", "-r", (Join-Path $Repo "requirements.txt"))
Run $Py @("-m", "pip", "install", "--progress-bar", "off", "--no-deps", "silero-vad==6.2.3")

# ---------------------------------------------------------------------------------------------
Step "whisper.cpp with Vulkan (pywhispercpp $PwcVersion)"
$site = Join-Path $Venv "Lib\site-packages"
if (Get-ChildItem $site -Filter "ggml-vulkan*.dll" -ErrorAction SilentlyContinue) {
    Write-Host "Already built with Vulkan."
} else {
    Run $Py @("-m", "pip", "install", "--progress-bar", "off", "setuptools", "wheel", "setuptools-scm>=8", "repairwheel")
    New-Item -ItemType Directory -Force $BuildRoot | Out-Null
    $src = Join-Path $BuildRoot "pwc"
    if (Test-Path $src) { Remove-Item -Recurse -Force $src }
    $env:SETUPTOOLS_SCM_PRETEND_VERSION = $PwcVersion   # the sdist has no git metadata
    Run $Py @("-m", "pip", "download", "--no-binary", "pywhispercpp", "--no-deps", "--no-build-isolation",
              "--progress-bar", "off", "-d", $BuildRoot, "pywhispercpp==$PwcVersion")
    Run "tar" @("-xzf", (Join-Path $BuildRoot "pywhispercpp-$PwcVersion.tar.gz"), "-C", $BuildRoot)
    Rename-Item (Join-Path $BuildRoot "pywhispercpp-$PwcVersion") $src
    # The Vulkan shader generator nests ~200 characters deep inside the build folder, past Windows'
    # 260-character path limit. Build on a temporary drive letter with short build folders.
    $letter = @("W", "V", "U", "T", "S", "R", "Q", "P") | Where-Object { -not (Test-Path "${_}:\") } | Select-Object -First 1
    if (-not $letter) { throw "No free drive letter for the build." }
    Add-Content -Path (Join-Path $src "setup.cfg") -Encoding ascii -Value "`n[build]`nbuild_temp = ${letter}:/t`nbuild_base = ${letter}:/b`n"
    Run "subst" @("${letter}:", $BuildRoot)
    try {
        $env:PATH = "$(Join-Path $Venv 'Scripts');$env:PATH"   # the build calls `repairwheel`
        $env:GGML_VULKAN = "1"
        $env:CMAKE_BUILD_PARALLEL_LEVEL = "$([Environment]::ProcessorCount)"
        Run $Py @("-m", "pip", "install", "--no-build-isolation", "--no-deps", "--force-reinstall", "${letter}:\pwc")
    } finally {
        & subst "${letter}:" /D | Out-Null
        Remove-Item Env:GGML_VULKAN -ErrorAction SilentlyContinue
    }
    if (-not (Get-ChildItem $site -Filter "ggml-vulkan*.dll")) { throw "pywhispercpp was built without the Vulkan backend." }
    Remove-Item -Recurse -Force $BuildRoot
}

# ---------------------------------------------------------------------------------------------
# Downloads resume where they stopped and are retried until the size matches the server's;
# school Wi-Fi resets long downloads.
function Download([string]$Url, [string]$Dest) {
    $head = & curl.exe -sIL --fail $Url
    if ($LASTEXITCODE -ne 0) { throw "Could not reach $Url" }
    $expected = [int64](($head | Select-String -Pattern "^content-length:\s*(\d+)" | Select-Object -Last 1).Matches[0].Groups[1].Value)
    $have = if (Test-Path $Dest) { (Get-Item $Dest).Length } else { 0 }
    if ($have -eq $expected) { Write-Host "Already downloaded ($([math]::Round($expected / 1MB)) MB)."; return }
    if ($have -gt $expected) { throw "$Dest is larger than the file on the server; delete it and run setup again." }
    Write-Host "Downloading $([math]::Round($expected / 1MB)) MB (have $([math]::Round($have / 1MB)) MB)..."
    for ($i = 1; $i -le 30; $i++) {
        & curl.exe -L --fail --retry 8 --retry-all-errors --retry-delay 5 -C - -o $Dest $Url
        if ((Get-Item $Dest -ErrorAction SilentlyContinue).Length -eq $expected) { return }
        Write-Host "Connection dropped; resuming (attempt $i)..."
        Start-Sleep -Seconds 5
    }
    throw "Download incomplete: $((Get-Item $Dest).Length) of $expected bytes. Run setup again to resume."
}

$modelsDir = Join-Path $Repo (ConfigValue "models_dir")
New-Item -ItemType Directory -Force $modelsDir | Out-Null
$models = @(
    @{ What = "Whisper model"; File = "ggml-$(ConfigValue 'asr_model').bin";
       Url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$(ConfigValue 'asr_model').bin" },
    @{ What = "Speaker model"; File = (ConfigValue "speaker_model");
       Url = "https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM/resolve/main/voxceleb_resnet34_LM.onnx" },
    @{ What = "Qwen model (Obsidian notes)"; File = (ConfigValue "llm_model");
       Url = "https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/$(ConfigValue 'llm_model')" }
)
foreach ($m in $models) {
    Step "$($m.What): $($m.File)"
    if ($SkipModel) { Write-Host "Skipped (-SkipModel)."; continue }
    Download $m.Url (Join-Path $modelsDir $m.File)
}

# ---------------------------------------------------------------------------------------------
$LlamaBuild = "b12068"
$llamaServer = (ConfigValue "llama_server").Replace("~", $env:USERPROFILE).Replace("/", "\")
Step "llama.cpp $LlamaBuild for Vulkan (runs Qwen on the GPU)"
if (Test-Path $llamaServer) {
    Write-Host "Already installed: $llamaServer"
} else {
    $llamaDir = Split-Path $llamaServer
    New-Item -ItemType Directory -Force $llamaDir | Out-Null
    $zip = Join-Path (Split-Path $llamaDir) "llama-$LlamaBuild-bin-win-vulkan-x64.zip"
    Download "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaBuild/llama-$LlamaBuild-bin-win-vulkan-x64.zip" $zip
    Expand-Archive -Path $zip -DestinationPath $llamaDir -Force
    Remove-Item $zip
    if (-not (Test-Path $llamaServer)) { throw "llama-server.exe is not where config.yaml expects it: $llamaServer" }
    & $llamaServer --list-devices
}

# ---------------------------------------------------------------------------------------------
Step "App shortcuts"
$icon = Join-Path $Venv "lecturelive.ico"
Push-Location $Repo
try { Run $Py @("-m", "lecturelive", "--icon", $icon) } finally { Pop-Location }
$shell = New-Object -ComObject WScript.Shell
$places = @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))
foreach ($dir in $places) {
    $lnk = $shell.CreateShortcut((Join-Path $dir "Lecture Live.lnk"))
    $lnk.TargetPath = $PyW
    $lnk.Arguments = "-m lecturelive"
    $lnk.WorkingDirectory = $Repo
    $lnk.IconLocation = $icon
    $lnk.Description = "Live lecture transcription (Whisper on the GPU, offline)"
    $lnk.Save()
    Write-Host "Shortcut: $(Join-Path $dir 'Lecture Live.lnk')"
}

Write-Host "`nDone. Start Lecture Live from the desktop or Start menu, or run .\run.cmd" -ForegroundColor Green
