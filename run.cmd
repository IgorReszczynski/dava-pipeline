@echo off
rem Start Lecture Live from a terminal (errors print here as well as in the window).
cd /d "%~dp0"
"%USERPROFILE%\.venvs\lecture-live\Scripts\python.exe" -m lecturelive %*
