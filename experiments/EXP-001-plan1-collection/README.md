# EXP-001 — Recolección piloto de CoinRun

## Objetivo

Verificar en CPU que el recolector guarda episodios HDF5 completos, alineados y reproducibles para políticas random, sticky y scripted, y habilitar captura humana con el teclado oficial de Procgen.

## Hipótesis

1. El formato por episodio conserva frames, acciones, rewards, dones y metadata sin pérdida.
2. Una mezcla piloto de políticas produce al menos una victoria y un terminal no ganador.
3. El mismo estado inicial y las mismas acciones reproducen exactamente los frames guardados.
4. `easy` puede aumentar victorias; esta hipótesis se contrasta, no se asume.

## Datos

Los HDF5 se encuentran bajo `data/raw/` y no se versionan. Este expediente contiene comandos, métricas, logs y una muestra visual. Los dos episodios `gui_smoke_idle` no son datos humanos válidos.
