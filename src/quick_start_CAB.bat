@echo off
echo プログラム開始


SET VIRTUAL_ENV_NAME="CAB"

REM 仮想環境をactivate
@echo on
call activate %VIRTUAL_ENV_NAME%

call python CAB.py