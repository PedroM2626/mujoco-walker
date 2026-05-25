# MuJoCo MPC Walker

Este projeto integra o `walker_ragdoll` como uma Custom Task dentro do framework oficial **MuJoCo MPC (MJPC)** da Google DeepMind, além de disponibilizar todos os modelos padrão do MJPC para comparação.

Como o MJPC é um framework C++, você precisará compilá-lo nativamente no Windows para usar a interface gráfica interativa (onde você pode visualizar as trajetórias sendo planejadas em tempo real, mudar modos e arrastar alvos).

## Pré-requisitos (Windows)

Para compilar, você precisa ter instalados:
1. **CMake** (versão 3.16 ou superior). Pode ser baixado em [cmake.org](https://cmake.org/download/) ou via `winget install cmake`.
2. **Visual Studio Build Tools 2022** (com suporte a "Desenvolvimento para desktop com C++").

## Como Compilar e Rodar

Para facilitar o processo, o projeto possui dois scripts de automação:

1. **Compilar:** Abra um terminal (PowerShell ou CMD) na pasta `mujoco_mpc_walker` e execute:
   ```bash
   .\build.bat
   ```
   Este script inicializa o ambiente de compilação do Visual Studio e compila o projeto em modo Release.

2. **Rodar:** Para abrir a janela de visualização do MuJoCo MPC, execute:
   ```bash
   .\run.bat
   ```
   Este script configura o caminho dos DLLs do MuJoCo, define as variáveis de ambiente necessárias para encontrar as definições de tarefas e inicia o executável `walker_mpc.exe`.

## Usando a Interface
Ao abrir a interface gráfica do MuJoCo MPC:
- No painel lateral, você verá uma lista de tarefas (Tasks). Você pode selecionar a sua custom task `Walker Ragdoll` ou tarefas clássicas do MJPC como `Walker`, `Humanoid Stand`, `Humanoid Walk` e `Acrobot`.
- O solver (Predictive Sampling) calculará a postura, controle e posições.
- Você pode ajustar os pesos dos **Resíduos** em tempo real para focar mais em Velocidade, Postura ou Altura.
- Você pode segurar `Shift` e usar o botão direito do mouse para mover/mudar a posição de alvos e interagir diretamente com a simulação.
