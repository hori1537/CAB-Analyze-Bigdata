@echo off
echo プログラム開始


echo graphvizの存在確認
SET filename_graphviz="%CD%\release\bin\dot.exe"

IF EXIST %filename_graphviz% (GOTO GRAPHVIZ_EXIST) ELSE GOTO GRAPHVIZ_NOT_EXIST

:GRAPHVIZ_EXIST

echo "graphvizがインストールされています"
GOTO GRAPHVIZ_END

:GRAPHVIZ_NOT_EXIST
ECHO "graphvizが存在しません。ダウンロードします"

echo graphvizのダウンロード
bitsadmin /transfer doanload_graphviz /PRIORITY FOREGROUND https://graphviz.gitlab.io/_pages/Download/windows/graphviz-2.38.zip %CD%\graphviz-2.38.zip

echo graphvizを解凍します
cd %~dp0
.\7za\7za.exe x graphviz-2.38.zip
echo Copyright (C) 2019 Igor Pavlov. OSDN Project translated.
TIMEOUT /T 3

REM ダウンロードした7zは削除しない
REM rd /s /q %homedrive%\7z
TIMEOUT /T 3


echo ダウンロードしたgraphviz-2.38.zipを削除します
del /s /q graphviz-2.38.zip
TIMEOUT /T 3

GOTO GRAPHVIZ_END


:GRAPHVIZ_END

REM start_programのあるフォルダに戻る
cd /d %~dp0

REM condaが実行できるか確認

call conda -V

if %errorlevel% neq 0 ( 
echo condaのPATHが通っていません
echo Anaconda プロンプトを起動し、start.batを実行してください
echo 10秒後にプログラムを終了します
TIMEOUT /T 10
exit /B
) else (
echo condaコマンドは正常に実行されました
)

SET VIRTUAL_ENV_NAME="CAB"

REM 仮想環境をactivate
@echo on
call activate %VIRTUAL_ENV_NAME%
@echo off

if %errorlevel% neq 0 (
echo 仮想環境のactivate失敗
echo 仮想環境の作成開始
@echo on
echo Y | call conda create -n %VIRTUAL_ENV_NAME% python=3.6
call activate %VIRTUAL_ENV_NAME%
) else (
@echo off
echo 仮想環境をactivateしました
)

@echo on
REM 必要なライブラリをpip install
call pip install -r requirements.txt

REM CAB.pyの更新
bitsadmin /transfer CAB.pyの最新ファイルをダウンロードします /PRIORITY FOREGROUND https://github.com/hori1537/CAB-Analyze-Bigdata/archive/master.zip %~dp0\master.zip
echo S | 7za\7za.exe x %~dp0\master.zip
echo Copyright (C) 2019 Igor Pavlov. OSDN Project translated.

cd %~dp0
copy CAB-Analyze-Bigdata-master\CAB.py CAB_new.py

echo 一時ファイルを削除します
rd /s /q CAB-Analyze-Bigdata-master
del master.zip
echo 一時ファイルを削除しました

TIMEOUT /T 3

SET UPDATE=FALSE
SET ANSWER=FALSE

echo N | comp CAB.py CAB_new.py

if %errorlevel% neq 0 (
echo CAB.pyの新しいバージョンがあります
SET /P ANSWER="CAB.pyを更新します。よろしいですか (Y/N)？"
)

IF /i {%ANSWER%}=={y} SET UPDATE=TRUE
IF /i {%ANSWER%}=={yes} SET UPDATE=TRUE

IF %UPDATE%==TRUE (
echo 処理開始
echo CAB.pyを新しいバージョンに更新します
copy CAB_new.py CAB.py
echo CAB_new.pyを削除します
TIMEOUT /T 3
cd %~dp0
del CAB_new.py

) else (
echo バージョンを更新しません 
echo CAB_new.pyを削除します
TIMEOUT /T 3
cd %~dp0
del CAB_new.py

)

@echo on

call python CAB.py