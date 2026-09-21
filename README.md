# genai-2026
Repository for the generative AI course(2026).

## Generated CSV/XLSX files

- `recordings_with_gitdate.csv` / `recordings_with_gitdate_comma.csv` / `recordings_with_gitdate.xlsx`: contains a `GitCommitDate` column (ISO 8601) for each recorded path.
- `recordings_with_fsdate.csv` / `recordings_with_fsdate_comma.csv` / `recordings_with_fsdate.xlsx`: contains a `LastWriteTime` filesystem timestamp for each recorded path.
- `recordings_latest_enhanced.csv` / `recordings_latest_enhanced_comma.csv` / `recordings_latest_enhanced.xlsx`: contains both `GitCommitDate` and `LastWriteTime` columns.

To regenerate the enhanced files locally, from the repo root run (PowerShell):

```powershell
# create enhanced pipe-delimited CSV
"Path|Url|File|GitCommitDate|LastWriteTime" | Out-File recordings_latest_enhanced.csv -Encoding utf8
Get-Content recordings_latest.csv | ForEach-Object {
	if ([string]::IsNullOrWhiteSpace($_)) { return }
	$parts = $_ -split '\|'
	$path = $parts[0]
	$gitDate = ''
	try { $gitDate = & git log -1 --format=%cI -- "$path" 2>$null } catch {}
	$fsDate = ''
	try { $fsDate = (Get-Item $path).LastWriteTime.ToString('o') } catch {}
	($parts -join '|') + '|' + $gitDate + '|' + $fsDate | Out-File recordings_latest_enhanced.csv -Append -Encoding utf8
}

# convert to comma CSV and XLSX using included scripts
Import-Csv -Delimiter '|' -Path recordings_latest_enhanced.csv | Export-Csv -NoTypeInformation -Path recordings_latest_enhanced_comma.csv -Encoding UTF8
E:/AI-2026/softwares/envs/genai-2026/python.exe csv_to_xlsx.py
```
