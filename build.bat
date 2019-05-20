@echo off
cd /d %~dp0
cd out
msbuild waifu2xcpp.sln /p:Configuration=Release /p:Platform=x64 -m
cd ..
