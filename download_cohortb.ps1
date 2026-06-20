$url    = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE224nnn/GSE224807/suppl/GSE224807_RAW.tar"
$outDir = "$env:USERPROFILE\Downloads"
$startTime = Get-Date
Write-Host "Starting download - 16 parallel connections..." -ForegroundColor Cyan

& aria2c -x 16 -s 16 --summary-interval=1 --console-log-level=notice -d $outDir -o "GSE224807_RAW.tar" $url 2>&1 | ForEach-Object {
    $line = "$_"
    if ($line -match '\((\d+)%\).*DL:([\d\.]+\S+).*ETA:(\S+)') {
        $pct     = [int]$Matches[1]
        $speed   = $Matches[2]
        $eta     = $Matches[3]
        $elapsed = (Get-Date) - $startTime
        $elStr   = "{0:hh\:mm\:ss}" -f $elapsed
        $filled  = [int]($pct / 100 * 40)
        $bar     = ("#" * $filled).PadRight(40, "-")
        Write-Host -NoNewline ("`r[{0}] {1,3}%  {2}/s  Elapsed:{3}  ETA:{4}   " -f $bar, $pct, $speed, $elStr, $eta)
    } elseif ($line -match 'Download complete') {
        Write-Host "`n$line"
    }
}

Write-Host ("`nDone! Total time: {0:hh\:mm\:ss}" -f ((Get-Date) - $startTime)) -ForegroundColor Green
