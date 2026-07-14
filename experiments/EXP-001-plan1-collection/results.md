# Resultados

## Evidencia

- 16/16 tests específicos de Plan 1 pasaron.
- 22/22 tests del repositorio pasaron.
- 20 episodios automáticos y 3,685 transiciones fueron validados offline y resimulados exactamente desde `initial_state`.
- `level_seed` y `source_policy` están completos en 20/20 episodios automáticos.
- Se observaron 5 victorias y 5 terminales no ganadores; la etiqueta `death` es un proxy explícito de death-or-fall.

## Easy frente a hard

Con la misma política scripted y cinco episodios por modo, `hard` produjo 5/5 victorias y `easy` 0/5; `easy` produjo 5/5 terminales no ganadores. Se mantiene `hard` como línea base para seed 0. Este resultado no demuestra que `hard` sea mejor en general.

## Fallos y limitaciones

1. Random y sticky se truncaron a 300 pasos sin terminales; necesitan barridos o mayor horizonte si se busca cobertura de eventos por sí solas.
2. La utilidad gráfica falló inicialmente por aliases EGL/GL ausentes; el workaround aislado está en `scripts/run_human_capture.sh`.
3. El intento de dos minutos tuvo cero acciones humanas distintas de no-op. Fue relabelado y cuarentenado, no contado como humano.
4. La implementación se ejecutó sobre el commit base indicado con working tree sucio; debe crearse un commit antes de la recolección definitiva.

## Decisión

**Go condicional.** El pipeline automático y el formato cumplen; antes de Plan 2 se requiere una sesión humana corta válida. Antes de la recolección definitiva también se deben firmar el alcance y versionar la implementación.
