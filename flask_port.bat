netsh advfirewall firewall add rule name="Flask Server" protocol=TCP dir=in localport=5000 action=allow
pause