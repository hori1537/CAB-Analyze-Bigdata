@echo off
echo �v���O�����J�n


SET VIRTUAL_ENV_NAME="CAB"

REM ���z����activate
@echo on
call activate %VIRTUAL_ENV_NAME%

call python CAB.py