@echo off
echo �v���O�����J�n


echo graphviz�̑��݊m�F
SET filename_graphviz="%CD%\release\bin\dot.exe"

IF EXIST %filename_graphviz% (GOTO GRAPHVIZ_EXIST) ELSE GOTO GRAPHVIZ_NOT_EXIST

:GRAPHVIZ_EXIST

echo "graphviz���C���X�g�[������Ă��܂�"
GOTO GRAPHVIZ_END

:GRAPHVIZ_NOT_EXIST
ECHO "graphviz�����݂��܂���B�_�E�����[�h���܂�"

echo graphviz�̃_�E�����[�h
bitsadmin /transfer doanload_graphviz /PRIORITY FOREGROUND https://graphviz.gitlab.io/_pages/Download/windows/graphviz-2.38.zip %CD%\graphviz-2.38.zip

echo graphviz���𓀂��܂�
cd %~dp0
.\7za\7za.exe x graphviz-2.38.zip
echo Copyright (C) 2019 Igor Pavlov. OSDN Project translated.
TIMEOUT /T 3

REM �_�E�����[�h����7z�͍폜���Ȃ�
REM rd /s /q %homedrive%\7z
TIMEOUT /T 3


echo �_�E�����[�h����graphviz-2.38.zip���폜���܂�
del /s /q graphviz-2.38.zip
TIMEOUT /T 3

GOTO GRAPHVIZ_END


:GRAPHVIZ_END

REM start_program�̂���t�H���_�ɖ߂�
cd /d %~dp0

REM conda�����s�ł��邩�m�F

call conda -V

if %errorlevel% neq 0 ( 
echo conda��PATH���ʂ��Ă��܂���
echo Anaconda �v�����v�g���N�����Astart.bat�����s���Ă�������
echo 10�b��Ƀv���O�������I�����܂�
TIMEOUT /T 10
exit /B
) else (
echo conda�R�}���h�͐���Ɏ��s����܂���
)

SET VIRTUAL_ENV_NAME="CAB"

REM ���z����activate
@echo on
call activate %VIRTUAL_ENV_NAME%
@echo off

if %errorlevel% neq 0 (
echo ���z����activate���s
echo ���z���̍쐬�J�n
@echo on
echo Y | call conda create -n %VIRTUAL_ENV_NAME% python=3.6
call activate %VIRTUAL_ENV_NAME%
) else (
@echo off
echo ���z����activate���܂���
)

@echo on
REM �K�v�ȃ��C�u������pip install
call pip install -r requirements.txt

REM CAB.py�̍X�V
bitsadmin /transfer CAB.py�̍ŐV�t�@�C�����_�E�����[�h���܂� /PRIORITY FOREGROUND https://github.com/hori1537/CAB-Analyze-Bigdata/archive/master.zip %~dp0\master.zip
echo S | 7za\7za.exe x %~dp0\master.zip
echo Copyright (C) 2019 Igor Pavlov. OSDN Project translated.

cd %~dp0
copy CAB-Analyze-Bigdata-master\CAB.py CAB_new.py

echo �ꎞ�t�@�C�����폜���܂�
rd /s /q CAB-Analyze-Bigdata-master
del master.zip
echo �ꎞ�t�@�C�����폜���܂���

TIMEOUT /T 3

SET UPDATE=FALSE
SET ANSWER=FALSE

echo N | comp CAB.py CAB_new.py

if %errorlevel% neq 0 (
echo CAB.py�̐V�����o�[�W����������܂�
SET /P ANSWER="CAB.py���X�V���܂��B��낵���ł��� (Y/N)�H"
)

IF /i {%ANSWER%}=={y} SET UPDATE=TRUE
IF /i {%ANSWER%}=={yes} SET UPDATE=TRUE

IF %UPDATE%==TRUE (
echo �����J�n
echo CAB.py��V�����o�[�W�����ɍX�V���܂�
copy CAB_new.py CAB.py
echo CAB_new.py���폜���܂�
TIMEOUT /T 3
cd %~dp0
del CAB_new.py

) else (
echo �o�[�W�������X�V���܂��� 
echo CAB_new.py���폜���܂�
TIMEOUT /T 3
cd %~dp0
del CAB_new.py

)

@echo on

call python CAB.py