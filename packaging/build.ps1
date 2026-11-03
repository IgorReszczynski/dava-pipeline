<#
Build Lecture Live as a Windows app.

  .\packaging\build.ps1           dist\Lecture Live\Lecture Live.exe and dist\Lecture-Live-win64.zip
                                  (models are not in the zip; the app downloads them on first start)
  .\packaging\build.ps1 -Install  also install it for this user: %LOCALAPPDATA%\Programs\Lecture Live,
                                  using the models already on this PC (hard links, no extra space),
                                  and point the desktop and Start-menu shortcuts at the .exe

Run setup.ps1 first: the build uses its Python environment and llama.cpp.
#>
param([switch]$Install)
$ErrorActionPreference = "Stop"

$Repo = Split-Path $PSScriptRoot
$Venv = Join-Path $env:USERPROFILE ".venvs\lecture-live"
$Py = Join-Path $Venv "Scripts\python.exe"
$Site = Join-Path $Venv "Lib\site-packages"
$Name = "Lecture Live"
$Dist = Join-Path $Repo "dist"
$App = Join-Path $Dist $Name
$Work = Join-Path $Repo "build"
$LlamaDir = Join-Path $env:USERPROFILE ".venvs\llama.cpp\b12068"

function Run([string]$Exe, [string[]]$ArgList) {
    & $Exe @ArgList
    if ($LASTEXITCODE -ne 0) { throw "$Exe failed with exit code $LASTEXITCODE" }
}

if (-not (Test-Path $Py)) { throw "Run setup.ps1 first (no Python environment at $Venv)." }
if (-not (Test-Path (Join-Path $LlamaDir "llama-server.exe"))) { throw "Run setup.ps1 first (no llama.cpp at $LlamaDir)." }
Run $Py @("-m", "pip", "install", "--progress-bar", "off", "pyinstaller==6.22.3")

New-Item -ItemType Directory -Force $Work | Out-Null
$icon = Join-Path $Work "lecturelive.ico"
Push-Location $Repo
try { Run $Py @("-m", "lecturelive", "--icon", $icon) } finally { Pop-Location }

# whisper.cpp's DLLs sit next to _pywhispercpp.pyd under names with a hash; bundle all of them
$binaries = Get-ChildItem $Site -Filter "*.dll" | ForEach-Object { "--add-binary=$($_.FullName);." }
$silero = Join-Path $Site "silero_vad\data\silero_vad.onnx"
$piArgs = @("-m", "PyInstaller", "--noconfirm", "--clean", "--windowed", "--name", $Name, "--icon", $icon,
            "--distpath", $Dist, "--workpath", (Join-Path $Work "pyinstaller"), "--specpath", $Work,
            "--paths", $Repo, "--add-data=$silero;silero_vad\data", "--exclude-module", "tkinter") + $binaries +
          @((Join-Path $PSScriptRoot "lecture_live.py"))
Run $Py $piArgs

Run $Py @((Join-Path $PSScriptRoot "prepare_dist.py"), "app", $App)
Copy-Item -Recurse -Force $LlamaDir (Join-Path $App "llama.cpp")
New-Item -ItemType Directory -Force (Join-Path $App "models") | Out-Null

$zip = Join-Path $Dist "Lecture-Live-win64.zip"
if (Test-Path $zip) { Remove-Item $zip }
Compress-Archive -Path $App -DestinationPath $zip
Write-Host ("Built: {0}  ({1:N0} MB zipped)" -f (Join-Path $App "$Name.exe"), ((Get-Item $zip).Length / 1MB))

if ($Install) {
    $Dest = Join-Path $env:LOCALAPPDATA "Programs\$Name"
    New-Item -ItemType Directory -Force $Dest | Out-Null
    # copy the program; keep the installed config.yaml, modules.json and models if they are there
    robocopy $App $Dest /E /NFL /NDL /NJH /NJS /XF config.yaml modules.json /XD models | Out-Null
    if ($LASTEXITCODE -ge 8) { throw "Copying the app to $Dest failed (robocopy $LASTEXITCODE)" }
    if (-not (Test-Path (Join-Path $Dest "config.yaml"))) { Copy-Item (Join-Path $App "config.yaml") $Dest }
    $models = Join-Path $Dest "models"
    New-Item -ItemType Directory -Force $models | Out-Null
    Get-ChildItem (Join-Path $Repo "models") -File | Where-Object { $_.Extension -ne ".part" } | ForEach-Object {
        $link = Join-Path $models $_.Name
        if (-not (Test-Path $link)) { New-Item -ItemType HardLink -Path $link -Target $_.FullName | Out-Null }
    }
    Run $Py @((Join-Path $PSScriptRoot "prepare_dist.py"), "install", $Dest)
    $exe = Join-Path $Dest "$Name.exe"
    $shell = New-Object -ComObject WScript.Shell
    foreach ($dir in @([Environment]::GetFolderPath("Desktop"), [Environment]::GetFolderPath("Programs"))) {
        $lnk = $shell.CreateShortcut((Join-Path $dir "$Name.lnk"))
        $lnk.TargetPath = $exe
        $lnk.Arguments = ""
        $lnk.WorkingDirectory = $Dest
        $lnk.IconLocation = $exe
        $lnk.Description = "Live lecture transcription (Whisper on the GPU, offline)"
        $lnk.Save()
    }
    Write-Host "Installed: $exe (shortcuts updated)"
}
