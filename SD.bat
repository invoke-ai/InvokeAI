set root=C:\Users\USERNAME\miniconda3
set sd=C:\stable-diffusion
set envName=ldm
:restart
call %root%\Scripts\activate.bat %root%
call cd %sd%
call conda activate %envName%
::can add startup parameters, check help, ex: --outdir
call %sd%\scripts\dream.py 
PAUSE
GOTO restart