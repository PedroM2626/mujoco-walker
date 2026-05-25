# MuJoCo MPC Walker

Este projeto integra o `walker_ragdoll` como uma Custom Task dentro do framework oficial **MuJoCo MPC (MJPC)** da Google DeepMind.

Como o MJPC é um framework C++, você precisará compilá-lo nativamente no Windows para usar a interface gráfica interativa (onde você pode visualizar as trajetórias sendo planejadas em tempo real, mudar modos e arrastar alvos).

## Pré-requisitos (Windows)

Para compilar, você precisa ter instalados:
1. **CMake** (versão 3.16 ou superior). Pode ser baixado em [cmake.org](https://cmake.org/download/) ou via `winget install cmake`.
2. **Visual Studio Build Tools 2022** (com suporte a "Desenvolvimento para desktop com C++").

## Como Compilar e Rodar

1. Abra um terminal (PowerShell ou CMD) na pasta `mujoco_mpc_walker`.
2. Configure o CMake executando:
   ```bash
   cmake -B build
   ```
3. Compile o projeto executando:
   ```bash
   cmake --build build --config Release
   ```
4. Após a compilação, o executável estará pronto. Rode o aplicativo:
   ```bash
   ./build/Release/walker_mpc.exe
   ```

## Usando a Interface
Ao abrir, você verá a GUI do MuJoCo. Na barra lateral esquerda (ou apertando a tecla `T` para abrir a janela de Tasks), você poderá interagir com o agente MPC:
- O painel mostrará o seu modelo `Walker Ragdoll`.
- O solver (Predictive Sampling) calculará a postura, controle e posições.
- Você pode ajustar os pesos dos **Resíduos** em tempo real para focar mais em Velocidade, Postura ou Altura.
