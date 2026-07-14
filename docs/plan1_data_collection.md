# Plan 1 — Diseño ejecutado y auditoría de recolección

## Convención

- **[HECHO VERIFICADO]**: medido localmente o sustentado por una fuente primaria.
- **[AFIRMACIÓN DEL AUTOR]**: resultado reportado por autores, no reproducido aquí.
- **[INFERENCIA DEL EQUIPO]**: conclusión propia a partir de la evidencia.
- **[HIPÓTESIS EXPERIMENTAL]**: decisión todavía pendiente de medición suficiente.

## Auditoría de Procgen 0.10.7

**[HECHO VERIFICADO]** La instalación incluye `procgen/interactive.py`. Su CLI real acepta `--level-seed`, no `--start-level` ni `--num-levels`; internamente traduce `--level-seed N` a `start_level=N, num_levels=1`.

**[HECHO VERIFICADO]** La captura reutiliza `ProcgenInteractive` y `gym3.Interactive` para ventana, temporización y consulta oficial `keys_to_act`. No reutiliza `VideoRecorderWrapper`, porque este guarda video pero no acciones, reward, done ni metadata HDF5.

**[HECHO VERIFICADO]** En CoinRun se observaron `D15[]`, RGB `uint8` de 64×64×3 y las claves de info `level_seed`, `prev_level_seed` y `prev_level_complete`. No se exponen posición del jugador, cámara ni una etiqueta separada de muerte/caída.

**[HECHO VERIFICADO]** `get_state`/`set_state` devolvieron y restauraron blobs idénticos. Los blobs iniciales medidos tuvieron 37,422 bytes en `hard` y 38,437 bytes en `easy` para seed 0. La documentación oficial también describe estas llamadas y confirma que `start_level` junto con `num_levels` fija el conjunto de niveles: [Procgen oficial](https://github.com/openai/procgen).

## Preflight gráfico

El primer arranque produjo el traceback completo conservado en `experiments/EXP-001-plan1-gui-preflight-initial-error.log`: ModernGL no encontraba `libEGL.so` ni `libGL.so`. WSL sí tenía `libEGL.so.1` y `libGL.so.1`, pero no los aliases de desarrollo, y `sudo` requería autenticación interactiva.

**[INGENIERÍA]** Se crearon aliases únicamente dentro de `.venv/lib` y se añadió esa carpeta a `LD_LIBRARY_PATH`. El proceso interactivo permaneció abierto cinco segundos y terminó por `timeout` con código 124, evidencia conservada en `experiments/EXP-001-plan1-gui-preflight-resolved.log`. `scripts/run_human_capture.sh` reproduce el workaround sin modificar el sistema.

## Formato HDF5 por episodio

**[INGENIERÍA]** Cada episodio se escribe primero como `.h5.partial` y se publica mediante rename atómico. HDF5 permite datasets tipados, metadata arbitraria, compresión LZF y lectura por episodio sin un archivo monolítico.

Datasets alineados por transición:

| Dataset | Tipo / forma | Semántica |
|---|---|---|
| `observations` | `uint8 [T,64,64,3]` | Frame anterior a la acción |
| `actions` | `uint8 [T]` | Acción aplicada, siempre 0–14 |
| `rewards` | `float32 [T]` | Reward devuelto después de esa acción |
| `dones` | `bool [T]` | `first` observado después de la acción |
| `truncateds` | `bool [T]` | Corte del recolector, distinto de terminal |
| `next_observations` | `uint8 [T,64,64,3]` | Frame siguiente solo si es válido |
| `next_observation_valid` | `bool [T]` | Falso en terminales para no guardar el reset como futuro |
| `events` | `uint8 [T]` | continue/victory/terminal_failure/truncated |
| `frame_indices`, `timestamps_ns` | `int64 [T]` | Índice y tiempo de captura |
| `level_seeds`, `source_policies`, `episode_ids` | `[T]` | Metadata lógica por transición |

Los atributos de episodio repiten `level_seed`, `source_policy`, `episode_id`, modo, versión de Procgen, commit y estado terminal. `initial_state` conserva el blob de Procgen para replay exacto.

**[INGENIERÍA]** Un `done` no ganador se registra como `terminal_failure` y como proxy `death=1`, con la definición explícita “death or fall no distinguible”. No se inventa una etiqueta separada de caída.

## Seeds preseleccionados

Smoke temporal, fuera del dataset, máximo 300 pasos en `hard`:

| Seed | Right+Up | Hop cycle | Estado |
|---:|---|---|---|
| 0 | victoria, 63 pasos | victoria, 79 pasos | Primario provisional |
| 1 | terminal no ganador, 64 | victoria, 148 | Candidato con diversidad útil |
| 2 | victoria, 47 | victoria, 54 | Candidato |
| 3 | victoria, 47 | victoria, 54 | Candidato |
| 4 | victoria, 111 | terminal no ganador, 92 | Candidato con peligro visible |

**[INFERENCIA DEL EQUIPO]** Las monedas fueron alcanzables por al menos una heurística y se observaron plataformas/obstáculos. Esto no sustituye la verificación humana de jugabilidad; los seeds 0–4 permanecen pendientes de confirmación manual y seed 0 se mantiene como primario para no cambiar silenciosamente Plan 0.

## Resultados automáticos del piloto

| Política / modo | Episodios | Frames | Victorias | Terminales no ganadores | Truncados |
|---|---:|---:|---:|---:|---:|
| random / hard | 5 | 1,500 | 0 | 0 | 5 |
| sticky p=0.7 / hard | 5 | 1,500 | 0 | 0 | 5 |
| scripted / hard | 5 | 395 | 5 | 0 | 0 |
| scripted / easy | 5 | 290 | 0 | 5 | 0 |
| **Total** | **20** | **3,685** | **5** | **5** | **10** |

Los 20 episodios tienen `level_seed` y `source_policy` completos (100%) y fueron verificados contra Procgen desde `initial_state`: 3,685/3,685 transiciones coincidieron.

**[REQUIERE VALIDACIÓN EXPERIMENTAL]** `easy` no mejoró las victorias para seed 0 y la heurística medida: 0/5 frente a 5/5 en `hard`. Se conserva `hard` como línea base; no se generaliza este resultado a otras políticas o seeds.

**[SUPUESTO]** `sticky p=0.7` fue solo el punto piloto. Deben compararse 0.5, 0.7 y 0.9 mediante cobertura de acciones y eventos antes de fijarlo.

## Cobertura humana

El preflight de dos minutos generó 1,755 transiciones pero registró cero acciones distintas de no-op. Esos archivos se relabelaron `gui_smoke_idle` y se movieron a `data/raw/gui_smoke_idle`; no cuentan como gameplay humano. La captura futura cuarentena automáticamente una sesión ociosa.

Por tanto, el piloto automático es válido pero el criterio de sesión humana corta sigue pendiente. Tampoco se han recolectado todavía los 40 minutos obligatorios.

## Contexto de escala de datos

**[AFIRMACIÓN DEL AUTOR]** Ha y Schmidhuber reportan 10,000 rollouts aleatorios para CarRacing y también para el experimento VizDoom: [World Models](https://worldmodels.github.io/).

**[AFIRMACIÓN DEL AUTOR]** GameNGen registra las sesiones durante el entrenamiento de un agente RL y usa frames y acciones para entrenar el modelo generativo: [Valevski et al., 2024](https://arxiv.org/abs/2408.14837).

**[INFERENCIA DEL EQUIPO]** Estas escalas no determinan un número óptimo para CoinRun, pero sí justifican medir cobertura y complementar los 40 minutos humanos con políticas automáticas.
