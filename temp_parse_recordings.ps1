$root = 'e:\AI-2026\algo-genai-2026'
$out = Join-Path $root 'recordings_latest.csv'
$entries = @()
Get-ChildItem -Path $root -Recurse -Filter '*recording*.txt' | Sort-Object FullName | ForEach-Object {
    $content = Get-Content -Path $_.FullName -ErrorAction SilentlyContinue
    $current = $null
    foreach ($line in $content) {
        $trim = $line.Trim()
        if ($trim -match 'https?://') {
            $current = [pscustomobject]@{
                File = $_.FullName
                Url = $trim
                Label = ''
            }
            $entries += $current
        } elseif ($current -and $trim -match '^\((.+\.mp4)\)$') {
            $current.Label = $matches[1]
        } elseif ($current -and $trim -and -not ($trim -match 'https?://') -and -not ($trim -match '^\(.+\)$')) {
            if (-not $current.Label) {
                $current.Label = $trim
            }
        }
    }
}
$entries | ForEach-Object { $_.File + '|' + $_.Url + '|' + $_.Label } | Set-Content -Path $out -Encoding utf8
Write-Output $out
