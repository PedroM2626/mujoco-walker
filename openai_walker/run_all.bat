@echo off
REM Pipeline completo Fase 4 (Walker2d-v5). Rode a partir de openai_walker\.
REM Etapas pesadas (AIRL/DT) podem levar horas; comente com REM para pular.
REM Pre-req: pip install -r ..\requirements.txt

python train_teacher.py
if errorlevel 1 exit /b %errorlevel%
python generate_dataset.py
if errorlevel 1 exit /b %errorlevel%

REM --- Offline puro ---
python train_bc.py
if errorlevel 1 exit /b %errorlevel%
python train_iql.py
if errorlevel 1 exit /b %errorlevel%
python train_cql.py
if errorlevel 1 exit /b %errorlevel%
python train_offline_bcq.py
if errorlevel 1 exit /b %errorlevel%
python train_offline_dt.py
if errorlevel 1 exit /b %errorlevel%
python train_extratrees.py
if errorlevel 1 exit /b %errorlevel%

REM --- IRL ---
python train_irl_gail.py
if errorlevel 1 exit /b %errorlevel%
python train_irl_airl.py
if errorlevel 1 exit /b %errorlevel%
python train_irl_maxent.py
if errorlevel 1 exit /b %errorlevel%
python train_irl_pqr.py
if errorlevel 1 exit /b %errorlevel%

REM --- Offline-to-online (fine-tuning SAC) ---
python train_bc_sac.py
if errorlevel 1 exit /b %errorlevel%
python train_bc_sac_regularized.py
if errorlevel 1 exit /b %errorlevel%
python train_bc_sac_constrained.py
if errorlevel 1 exit /b %errorlevel%
python train_iql_sac.py
if errorlevel 1 exit /b %errorlevel%
python train_cql_sac.py
if errorlevel 1 exit /b %errorlevel%

REM --- Corrida final (requer display; em servidor use evaluate_all.py) ---
python play_race.py > final_results.txt
