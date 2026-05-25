@echo off
set "PATH=%PATH%;C:\Program Files\Git\usr\bin"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64
"C:\Program Files\CMake\bin\cmake.exe" --build build --config Release
