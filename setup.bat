@echo off
echo ========================================
echo   MangaTL Setup
echo ========================================
echo.

REM Install engine dependencies
echo.
echo Installing engine dependencies...
cd tl-core
pip install -e .
pip install -r requirements.txt
cd ..

REM Install MangaTL dependencies
echo.
echo Installing MangaTL dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo   Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Download detector.onnx from HuggingFace
echo   2. Place it in models/detector.onnx
echo   3. Add API keys to .env (optional)
echo   4. Run: python run.py -i manga/ -o output/
echo.
pause
