$ErrorActionPreference = "Stop"

$destination = Join-Path $PSScriptRoot "..\Datasets_dir\Paderborn\archives"
New-Item -ItemType Directory -Force $destination | Out-Null

$files = @(
    @{
        Name = "K001.rar"
        Size = 173881721
        MD5 = "AF6DC58283D356CD438E7738A2D525B8"
    },
    @{
        Name = "KA01.rar"
        Size = 166571438
        MD5 = "5627EC9320199078205DB5325C1C2F84"
    },
    @{
        Name = "KI01.rar"
        Size = 175265779
        MD5 = "03429A4AB5E3768B7395D1654E55BEAD"
    }
)

foreach ($file in $files) {
    $target = Join-Path $destination $file.Name
    $url = "https://zenodo.org/api/records/15845309/files/$($file.Name)/content"
    curl.exe -L -C - --retry 8 --retry-all-errors --fail --output $target $url
    if ($LASTEXITCODE -ne 0) {
        throw "Download failed: $($file.Name)"
    }
    if ((Get-Item $target).Length -ne $file.Size) {
        throw "Unexpected file size: $($file.Name)"
    }
    $actual = (Get-FileHash $target -Algorithm MD5).Hash
    if ($actual -ne $file.MD5) {
        throw "MD5 mismatch: $($file.Name)"
    }
}

Write-Output "Paderborn archives downloaded and verified."
