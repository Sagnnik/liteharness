#Requires -Version 5.1

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$Package = 'ness-agent'
$CommandName = 'ness.exe'
$UvVersion = '0.12.3'
$UvInstallerUri = "https://releases.astral.sh/github/uv/releases/download/$UvVersion/uv-installer.ps1"
$UvInstallerSha256 = '7b84813e3fad9586da122e362d4dcba1e2e611664244d004bcfc32b2fdf10430'
$PyPiIndexUri = 'https://pypi.org/simple'
$TemporaryDirectory = $null

$ManagedEnvironmentVariables = @(
    'UV_CONFIG_FILE',
    'UV_DEFAULT_INDEX',
    'UV_EXTRA_INDEX_URL',
    'UV_FIND_LINKS',
    'UV_INDEX',
    'UV_INDEX_URL',
    'UV_INSECURE_HOST',
    'UV_INSTALL_DIR',
    'UV_NO_INDEX',
    'UV_NO_MODIFY_PATH',
    'UV_UNMANAGED_INSTALL'
)
$SavedEnvironment = @{}
foreach ($Name in $ManagedEnvironmentVariables) {
    $SavedEnvironment[$Name] = [Environment]::GetEnvironmentVariable($Name, 'Process')
}

function Write-Step {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host '==> ' -NoNewline -ForegroundColor White
    Write-Host $Message
}

function Write-Ready {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host '    ok  ' -NoNewline -ForegroundColor Green
    Write-Host $Message
}

function Write-WarningMessage {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host '    warning  ' -NoNewline -ForegroundColor Yellow
    Write-Host $Message -ForegroundColor Yellow
}

function Show-InstallLog {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ((Test-Path -LiteralPath $Path) -and (Get-Item -LiteralPath $Path).Length -gt 0) {
        Get-Content -LiteralPath $Path | ForEach-Object { Write-Host $_ }
    }
}

function Clear-ProcessEnvironmentVariable {
    param([Parameter(Mandatory = $true)][string]$Name)
    [Environment]::SetEnvironmentVariable($Name, $null, 'Process')
}

try {
    if ($env:OS -ne 'Windows_NT') {
        throw 'This installer supports Windows only.'
    }
    if ([string]::IsNullOrWhiteSpace($env:Path)) {
        throw 'PATH is not set. Set PATH, then run this installer again.'
    }

    $UserProfile = [Environment]::GetFolderPath([Environment+SpecialFolder]::UserProfile)
    if ([string]::IsNullOrWhiteSpace($UserProfile)) {
        throw 'The Windows user profile directory could not be determined.'
    }

    $ConfiguredUvInstallDir = [Environment]::GetEnvironmentVariable('UV_INSTALL_DIR', 'Process')
    if ([string]::IsNullOrWhiteSpace($ConfiguredUvInstallDir)) {
        $UvInstallDir = Join-Path $UserProfile '.local\bin'
    }
    else {
        $UvInstallDir = $ConfiguredUvInstallDir
    }
    if (-not [IO.Path]::IsPathRooted($UvInstallDir)) {
        throw "UV_INSTALL_DIR must be an absolute path: $UvInstallDir"
    }

    $TemporaryDirectory = Join-Path ([IO.Path]::GetTempPath()) ("ness-agent-install-" + [Guid]::NewGuid().ToString('N'))
    $null = New-Item -ItemType Directory -Path $TemporaryDirectory
    $InstallLog = Join-Path $TemporaryDirectory 'install.log'
    $UvInstallerPath = Join-Path $TemporaryDirectory 'uv-installer.ps1'

    Write-Host ''
    Write-Host 'NESS AGENT' -ForegroundColor White
    Write-Host 'own the loop' -ForegroundColor DarkGray
    Write-Host ''

    $UvCommand = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
    $ExpectedUvPath = Join-Path $UvInstallDir 'uv.exe'
    if ($null -ne $UvCommand) {
        $UvBin = $UvCommand.Source
        Write-Ready 'uv found'
    }
    elseif (Test-Path -LiteralPath $ExpectedUvPath -PathType Leaf) {
        $UvBin = $ExpectedUvPath
        Write-Ready "uv found at $UvBin"
    }
    else {
        Write-Step "Installing uv $UvVersion"

        $InstallerUri = [Uri]$UvInstallerUri
        if ($InstallerUri.Scheme -ne 'https') {
            throw "Refusing to download uv over a non-HTTPS URL: $UvInstallerUri"
        }
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -Uri $InstallerUri -OutFile $UvInstallerPath

        $ActualSha256 = (Get-FileHash -LiteralPath $UvInstallerPath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($ActualSha256 -ne $UvInstallerSha256) {
            throw "uv installer checksum mismatch. Expected $UvInstallerSha256, got $ActualSha256."
        }

        [Environment]::SetEnvironmentVariable('UV_INSTALL_DIR', $UvInstallDir, 'Process')
        [Environment]::SetEnvironmentVariable('UV_NO_MODIFY_PATH', '1', 'Process')
        Clear-ProcessEnvironmentVariable 'UV_UNMANAGED_INSTALL'

        $PowerShellExecutable = Join-Path $PSHOME 'powershell.exe'
        if (-not (Test-Path -LiteralPath $PowerShellExecutable -PathType Leaf)) {
            $PowerShellExecutable = Join-Path $PSHOME 'pwsh.exe'
        }
        if (-not (Test-Path -LiteralPath $PowerShellExecutable -PathType Leaf)) {
            throw 'Could not locate the current PowerShell executable.'
        }

        & $PowerShellExecutable -NoLogo -NoProfile -ExecutionPolicy Bypass -File $UvInstallerPath *> $InstallLog
        if ($LASTEXITCODE -ne 0) {
            Write-Host ''
            Write-Host 'uv installation failed.' -ForegroundColor Red
            Show-InstallLog $InstallLog
            throw "uv installer exited with code $LASTEXITCODE."
        }

        $UvBin = $ExpectedUvPath
        if (-not (Test-Path -LiteralPath $UvBin -PathType Leaf)) {
            throw "uv was installed, but its executable was not found at $UvBin"
        }
        Write-Ready 'uv installed'
    }

    Write-Step 'Installing or updating Ness Agent'

    foreach ($Name in @(
        'UV_CONFIG_FILE',
        'UV_DEFAULT_INDEX',
        'UV_EXTRA_INDEX_URL',
        'UV_FIND_LINKS',
        'UV_INDEX',
        'UV_INDEX_URL',
        'UV_INSECURE_HOST',
        'UV_NO_INDEX'
    )) {
        Clear-ProcessEnvironmentVariable $Name
    }

    & $UvBin tool install --python 3.12 --upgrade --no-config --default-index $PyPiIndexUri $Package *> $InstallLog
    if ($LASTEXITCODE -ne 0) {
        Write-Host ''
        Write-Host 'Ness Agent installation failed.' -ForegroundColor Red
        Show-InstallLog $InstallLog
        throw "uv exited with code $LASTEXITCODE."
    }

    $BinDirOutput = @(& $UvBin tool dir --bin --no-config 2>$null)
    if ($LASTEXITCODE -ne 0 -or $BinDirOutput.Count -eq 0) {
        throw 'The install completed, but the Ness executable directory could not be determined.'
    }
    $BinDir = [string]$BinDirOutput[$BinDirOutput.Count - 1]
    $NessBin = Join-Path $BinDir $CommandName
    if (-not (Test-Path -LiteralPath $NessBin -PathType Leaf)) {
        throw "The install completed, but ness.exe was not found at $NessBin"
    }

    $PathReady = $false
    foreach ($Entry in ($env:Path -split [IO.Path]::PathSeparator)) {
        if ([string]::Equals($Entry.TrimEnd('\'), $BinDir.TrimEnd('\'), [StringComparison]::OrdinalIgnoreCase)) {
            $PathReady = $true
            break
        }
    }

    $PathUpdateFailed = $false
    if (-not $PathReady) {
        Clear-Content -LiteralPath $InstallLog -ErrorAction SilentlyContinue
        & $UvBin tool update-shell --no-config *> $InstallLog
        if ($LASTEXITCODE -ne 0) {
            $PathUpdateFailed = $true
            Write-WarningMessage 'Could not update your Windows user PATH automatically.'
            Show-InstallLog $InstallLog
            Write-Host "    Ness was installed at: $NessBin" -ForegroundColor Yellow
        }
    }

    Write-Ready 'Ness Agent is up to date'
    Write-Host ''
    Write-Host 'Ready. ' -NoNewline -ForegroundColor White
    if ($PathReady) {
        Write-Host 'Run it with:'
        Write-Host ''
        Write-Host '  ness' -ForegroundColor White
    }
    elseif ($PathUpdateFailed) {
        Write-Host 'Run it now with:'
        Write-Host ''
        Write-Host "  & '$NessBin'" -ForegroundColor White
        Write-Host ''
        Write-Host 'Or add it to this PowerShell session:'
        Write-Host ''
        Write-Host "  `$env:Path = '$BinDir;' + `$env:Path" -ForegroundColor DarkGray
    }
    else {
        Write-Host 'Restart PowerShell, then run:'
        Write-Host ''
        Write-Host '  ness' -ForegroundColor White
        Write-Host ''
        Write-Host "Installed at $NessBin" -ForegroundColor DarkGray
    }
    Write-Host ''
}
catch {
    Write-Host ''
    Write-Host 'Installation failed.' -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    throw
}
finally {
    foreach ($Name in $ManagedEnvironmentVariables) {
        [Environment]::SetEnvironmentVariable($Name, $SavedEnvironment[$Name], 'Process')
    }
    if ($null -ne $TemporaryDirectory -and (Test-Path -LiteralPath $TemporaryDirectory)) {
        Remove-Item -LiteralPath $TemporaryDirectory -Recurse -Force -ErrorAction SilentlyContinue
    }
}
