@echo off
set "PATH=%PATH%;C:\Program Files\Git\usr\bin"
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" x64

cd "%~dp0"
if not exist build mkdir build
cd build

"C:\Program Files\CMake\bin\cmake.exe" .. -DCMAKE_BUILD_TYPE=Release
"C:\Program Files\CMake\bin\cmake.exe" --build . --config Release -j
