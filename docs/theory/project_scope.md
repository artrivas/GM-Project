# Alcance técnico y fundamento — Plan 0

## Convención de evidencia

- **[HECHO VERIFICADO]**: comprobado mediante una fuente primaria o una medición reproducible.
- **[AFIRMACIÓN DEL AUTOR]**: resultado o afirmación reportada por los autores, sin verificación independiente del equipo.
- **[INFERENCIA DEL EQUIPO]**: conclusión propia derivada de la evidencia disponible.
- **[HIPÓTESIS EXPERIMENTAL]**: proposición aún no resuelta que deberá medirse.

## Definición del sistema

**[INFERENCIA DEL EQUIPO]** El entregable se define como una simulación visual aprendida y condicionada por acciones, no como una política de control ni como una interfaz sobre el simulador original. Por eso la arquitectura inicial será un RSSM compacto con encoder, dinámica latente recurrente, decoder y heads de recompensa, terminación y evento; no incluirá actor ni critic.

**[HECHO VERIFICADO]** Ha y Schmidhuber describen un world model que aprende representaciones espaciales y temporales comprimidas de observaciones, y separan el modelo del controlador. Fuente primaria: [World Models, arXiv:1803.10122](https://arxiv.org/abs/1803.10122).

**[AFIRMACIÓN DEL AUTOR]** Los autores de GameNGen reportan simulación interactiva de DOOM a más de 20 FPS en una TPU y generación autoregresiva estabilizada con aumentos de condicionamiento. También indican que recolectaron las sesiones de entrenamiento de un agente RL para entrenar el modelo generativo. Fuente primaria: [Diffusion Models Are Real-Time Game Engines, arXiv:2408.14837](https://arxiv.org/abs/2408.14837).

**[INFERENCIA DEL EQUIPO]** Esos resultados no justifican prometer generalización procedural ni calidad tipo GameNGen con 40 minutos humanos: el entorno, arquitectura, cómputo y escala de datos son distintos. No se afirma aquí una razón cuantitativa exacta de “2–3 órdenes de magnitud” porque no se ha hecho una comparación normalizada de ambos datasets.

## Alcance obligatorio

- Observaciones RGB de 64×64 y las 15 acciones discretas de CoinRun.
- Control por teclado y generación de cada frame del rollout mediante el modelo aprendido.
- Objetivos medibles de ≥10 FPS y ≤100 ms por frame.
- Movimiento, salto, plataformas, enemigo, obstáculos, moneda, muerte/caída, victoria y reinicio.
- Progresión de seeds: uno fijo → tres → cinco → evaluación opcional en no vistos.
- Uso obligatorio de aproximadamente 40 minutos de gameplay humano, complementable con políticas automáticas.
- Separación de train/validation/test por episodios y seeds.

## Fuera de alcance inicial

- Actor, critic, PPO, Dreamer completo o aprendizaje por imaginación.
- Transformers temporales grandes, difusión pesada o modelos fundacionales.
- Promesa de generalización inmediata a niveles procedurales arbitrarios.
- Uso del simulador Procgen para producir frames durante la demo del world model.

## Decisiones de entorno

**[HECHO VERIFICADO]** La documentación oficial indica que `start_level` y `num_levels` especifican completamente el conjunto de niveles posibles. Con `num_levels=1`, `start_level` fija el único nivel. También declara que `distribution_mode="hard"` es el valor por defecto, que `"easy"` reduce los pasos necesarios para resolver los juegos, que Procgen no utiliza GPU y que se debe fijar una versión exacta con `==`. Fuente primaria: [repositorio oficial de Procgen](https://github.com/openai/procgen).

**[HECHO VERIFICADO]** PyPI publica wheels de `procgen==0.10.7` para CPython 3.7–3.10 y Linux x86‑64, pero no para Python 3.14. Fuente primaria: [Procgen 0.10.7 en PyPI](https://pypi.org/project/procgen/0.10.7/).

**[INGENIERÍA]** La línea base local será Python 3.10, `procgen==0.10.7`, un seed inicial 0, `num_levels=1`, `distribution_mode="hard"` y `num_threads=0`. La última opción evita problemas de fork durante pruebas y no cambia la semántica del nivel.

**[HIPÓTESIS EXPERIMENTAL]** En Plan 1 se medirá si `distribution_mode="easy"` aumenta suficientemente la cobertura de victorias sin empobrecer de forma inaceptable la variedad visual y de eventos. Hasta entonces no sustituye la línea base `"hard"`.

## Criterios generales Go/No-Go

- **Go:** entorno reproducible en Python 3.10, smoke de CoinRun aprobado y acuerdo firmado.
- **Go condicional:** smoke aprobado pero acuerdo aún sin firma, o falta GPU local para Plan 3+ con acceso remoto aún por confirmar.
- **No-Go:** Procgen no puede crear y avanzar un CoinRun en un entorno soportado; no hay ruta aprobada para los datos humanos; o se exige generalización no vista como requisito mínimo sin ampliar recursos y calendario.

## Riesgos abiertos de Plan 0

1. El Python predeterminado de WSL puede estar fuera del rango de wheels de Procgen.
2. No hay dataset parcial en el repositorio auditado.
3. No hay GPU NVIDIA/CUDA visible localmente.
4. El acuerdo requiere firmas humanas; Codex no puede firmar en nombre del equipo.
5. Las metas de FPS, latencia, estabilidad y controlabilidad siguen siendo objetivos, no resultados.
