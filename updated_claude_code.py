$connectTestResult = Test-NetConnection -ComputerName ctonpcimrastracc.file.core.windows.net -Port 445
if ($connectTestResult.TcpTestSucceeded) {
    # Save the password so the drive will persist on reboot
    cmd.exe /C "cmdkey /add:`"ctonpcimrastracc.file.core.windows.net`" /user:`"localhost\ctonpcimrastracc`" /pass:`"QI+OWdJJR1opISHQHkquzIy4tkNvIY4TJ9jv8AdtYIBC3WQRoC1EmctpTVHXSuDClSqseQ7VlFaJ+ASt6+o3vg==`""
    # Mount the drive
    New-PSDrive -Name Y -PSProvider FileSystem -Root "\\ctonpcimrastracc.file.core.windows.net\mrapermanentstorage" -Persist
} else {
    Write-Error -Message "Unable to reach the Azure storage account via port 445. Check to make sure your organization or ISP is not blocking port 445, or use Azure P2S VPN, Azure S2S VPN, or Express Route to tunnel SMB traffic over a different port."
}
