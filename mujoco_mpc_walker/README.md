# MuJoCo MPC Walker - Behavioral Cloning

Treinamento de uma rede neural student que aprende a imitar o planejador MPC (professor) para controlar um humanoide walker.

## Fluxo de Trabalho

```
1. Coleta de dados do professor  ->  collect_data.py
2. Treinamento do student        ->  train.py
3. Visualizacao do student       ->  play.py
```

## Requisitos

### Build do C++ (necessario uma vez)

```powershell
cd mujoco_mpc_walker
.\build.bat
```

### Dependencias Python

```bash
pip install -r requirements.txt
```

## Como Usar

### 1. Coletar dados do professor MPC

```bash
python collect_data.py
```

Isso vai:
- Abrir o **mjpc.exe** com interface grafica completa para voce ver o MPC em acao
- Rodar o **generate_dataset.exe** em paralelo (headless) coletando pares (estado, acao)
- Salvar os dados em `build/dataset.csv`

Opcoes:
```bash
python collect_data.py --steps 30000   # coletar mais amostras (padrao: 15000)
python collect_data.py --no-gui        # apenas coleta, sem janela visual
```

### 2. Treinar o student

```bash
python train.py
```

- Treina uma rede MLP com os dados coletados
- Usa MLflow para registrar metricas, hiperparametros e artefatos
- Salva o modelo em `teacher_model.pt` e o scaler em `scaler.pkl`

Visualizar experimentos no MLflow:
```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

### 3. Visualizar o student

```bash
python play.py
```

- Abre o viewer do MuJoCo com a rede neural controlando o robo
- Voce pode arrastar o alvo (bolinha vermelha) com Ctrl + Botao Direito

## Arquitetura

```
Estado (entrada) = [rel_tx, rel_ty, qpos[2:], qvel]   (47 dimensoes)
                          |
               MLP: 256 -> 256 -> 256
                          |
         Acoes de controle (17 motores)
```

A entrada usa **posicao relativa do alvo** (invariante de translacao), descartando as posicoes absolutas X e Y do torso.

## Estrutura de Arquivos

```
mujoco_mpc_walker/
├── collect_data.py     # Coleta dados do MPC com visualizacao
├── train.py            # Treina o student (Behavioral Cloning)
├── play.py             # Visualiza o student treinado
├── generate_data.cc    # Coletor headless em C++ (chamado pelo collect_data.py)
├── walker_task.cc/h    # Definicao da tarefa customizada Walker Ragdoll
├── task_walker.xml     # Configuracao do agente MPC (planner, horizonte, etc.)
├── CMakeLists.txt      # Build do C++
├── build.bat           # Script de compilacao para Windows
├── build/
│   ├── bin/mjpc.exe           # GUI do MuJoCo MPC
│   ├── generate_dataset.exe   # Coletor headless
│   └── dataset.csv            # Dados coletados (gerado pelo collect_data.py)
├── teacher_model.pt    # Modelo student treinado
└── scaler.pkl          # Normalizador dos dados de entrada
```

## MLOps

Cada run de treinamento e registrada no MLflow com:
- Hiperparametros: `learning_rate`, `batch_size`, `epochs`, `dataset_size`
- Metricas: `train_loss`, `val_loss` por epoca
- Artefatos: `teacher_model.pt`

## Docker

```bash
docker build -t mujoco-walker .
docker run --rm mujoco-walker python train.py
```
