# Diseño del dataset de CoinRun

## Categorías de evidencia

- **[HECHO VERIFICADO]**: fuente primaria o medición reproducible.
- **[AFIRMACIÓN DEL AUTOR]**: resultado reportado por un paper.
- **[INFERENCIA DEL EQUIPO]**: decisión derivada de la evidencia.
- **[HIPÓTESIS EXPERIMENTAL]**: pendiente de medición.

## Unidad de almacenamiento y partición

**[INFERENCIA DEL EQUIPO]** El episodio es la unidad indivisible de escritura, auditoría y split. Ningún frame de un episodio puede aparecer en más de una partición. En evaluación de generalización, seeds completos también deben quedar fuera de train.

**[HECHO VERIFICADO]** Procgen documenta que `start_level` y `num_levels` especifican completamente el conjunto de niveles; por ello estos valores forman parte de la metadata obligatoria. Fuente: [repositorio oficial de Procgen](https://github.com/openai/procgen).

### Split de Plan 2

**[INGENIERÍA]** Con tres o más seeds, el split de evaluación reservará seeds completos. En el piloto actual solo existe el seed 0, así que Plan 2 divide por episodio dentro de ese seed y lo declara explícitamente como limitación temporal. Esta partición no mide generalización procedural.

**[INGENIERÍA]** El contenido de los datasets de transición se resume con SHA-256. Episodios con la misma huella se asignan como un solo grupo indivisible, aunque tengan IDs distintos. Así se evita que una trayectoria scripted repetida aparezca a la vez en entrenamiento y evaluación.

**[REQUIERE VALIDACIÓN EXPERIMENTAL]** Cada split debe contener al menos una victoria única y una falla terminal única. Los conteos físicos no bastan: copias byte a byte de una misma trayectoria cuentan una sola vez para esta puerta de calidad.

**[SUPUESTO temporal]** Los manifests de un split que no supere esa puerta pueden conservarse para diagnosticar el fallo, pero se marcan `DO_NOT_USE` y no autorizan continuar al modelado.

## Alineación temporal

La transición lógica `t` es:

```text
observation[t] --action[t]--> reward[t], done[t], next_observation[t]
```

**[HECHO VERIFICADO]** En Gym3, después de un terminal, `observe()` entrega el frame inicial del episodio siguiente con `first=True`. Por tanto, el recolector guarda `next_observation_valid=False` y ceros en el next-frame terminal para evitar fuga entre episodios. Reward y done permanecen asociados a la acción que acaba de ejecutarse.

## Metadata mínima

Cada transición expone frame, acción, reward, done, episode ID, level seed, source policy e índice/timestamp. Los atributos constantes se repiten físicamente por episodio y lógicamente por transición mediante datasets de cobertura verificable.

**[HIPÓTESIS EXPERIMENTAL]** La posición del jugador, cámara y distinción muerte/caída podrían inferirse posteriormente mediante anotación o visión. Procgen 0.10.7 no las expone en el info de CoinRun auditado, de modo que no se inventan en Plan 1.

## Mezcla de políticas

**[AFIRMACIÓN DEL AUTOR]** World Models reporta 10,000 rollouts de política aleatoria en sus experimentos visuales: [Ha y Schmidhuber, 2018](https://worldmodels.github.io/).

**[AFIRMACIÓN DEL AUTOR]** GameNGen registra sesiones de un agente durante su entrenamiento antes de entrenar el generador condicionado por acciones: [Valevski et al., 2024](https://arxiv.org/abs/2408.14837).

**[INFERENCIA DEL EQUIPO]** El dataset CoinRun combinará gameplay humano obligatorio con random, sticky y scripted. Una política entrenada solo se añadirá si existe un checkpoint real y documentado; no se implementará PPO como parte del world model.

**[HIPÓTESIS EXPERIMENTAL]** La mezcla, el parámetro sticky y el uso de `easy` se decidirán según cobertura medida de acciones, victorias, terminales, enemigos y saltos, no por proporciones asumidas.
