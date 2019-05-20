set CurrDir=%~dp0
set VS_VER="15.0"
for /f "skip=2 tokens=1,2*" %%A in ('REG query "HKEY_LOCAL_MACHINE\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\SxS\VS7" /v %VS_VER% 2^>nul') DO set VS150PATH=%%C
cd /d "%VS150PATH%VC\Auxiliary\Build"
start vcvarsall.bat amd64