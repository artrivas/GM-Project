# Acuerdo de alcance — World Model interactivo de CoinRun

**Versión:** 1.0

**Fecha de preparación:** 2026-07-13

**Estado:** pendiente de firma del equipo

**Preparado por:** Codex, auditoría técnica de Plan 0

## Compromiso de producto

El proyecto desarrollará un world model visual interactivo para Procgen CoinRun, condicionado por las acciones del usuario y capaz de generar observaciones RGB de 64×64 a un mínimo medido de 10 FPS y con una latencia por frame no mayor de 100 ms. Durante el rollout generado, el siguiente frame procederá exclusivamente del modelo aprendido: queda prohibido utilizar `env.step()`, `env.act()` o cualquier transición equivalente del simulador Procgen.

El alcance mínimo comprenderá uno o más niveles previamente fijados mediante seeds conocidas (`start_level` junto con `num_levels=1`). El sistema deberá representar de manera controlable y persistente movimiento, salto, plataformas, al menos un tipo de enemigo, obstáculos, muerte o caída, moneda, victoria y reinicio.

La generalización a niveles procedurales no observados durante el entrenamiento será un objetivo adicional y no un requisito mínimo del entregable.

Los 40 minutos de gameplay humano establecidos por el curso se utilizarán obligatoriamente. Podrán complementarse con trayectorias automáticas producidas por políticas aleatorias, sticky, scripted o entrenadas para mejorar la cobertura de acciones y eventos poco frecuentes. El dataset se dividirá por episodios y seeds, nunca mediante una partición aleatoria de frames pertenecientes a una misma trayectoria.

El desarrollo seguirá esta progresión obligatoria:

1. Sobreajuste controlado sobre un seed y un conjunto pequeño de frames.
2. Validación contrafactual de que las acciones modifican la predicción.
3. Rollouts autoregresivos progresivamente más largos.
4. Ampliación a tres seeds y después a cinco seeds.
5. Evaluación opcional en seeds no observadas.

El criterio principal de éxito será una simulación aprendida estable, controlable y reproducible, no alcanzar resultados estado del arte. La arquitectura inicial será un RSSM compacto sin actor, critic ni entrenamiento por imaginación.

## MVP defendible si el tiempo se agota

El MVP aceptable será una demo de un seed conocido que cargue un checkpoint real, acepte control por teclado y produzca por el modelo movimiento, salto, un enemigo, muerte, moneda, victoria y reinicio a 64×64, ≥10 FPS y ≤100 ms por frame. Estas cifras deberán medirse en el hardware declarado. Un mock, una reconstrucción teacher-forced o una demo que consulte Procgen durante el rollout no satisfacen el MVP.

## Límites y reglas de evidencia

- Procgen solo podrá avanzar durante recolección de datos o smoke tests explícitamente identificados.
- No se ampliará a varios seeds antes de demostrar sobreajuste y controlabilidad en uno.
- No se declararán métricas, FPS, estabilidad ni determinismo que no hayan sido medidos.
- La opción `distribution_mode="hard"` será la línea base. El uso de `"easy"` requiere un experimento documentado y no se asumirá silenciosamente.
- La falta de datos, eventos raros, hardware o dependencias producirá un Go condicional o No-Go explícito.

## Aprobación

La firma confirma que el equipo acepta este alcance mínimo y que la generalización procedural no observada es adicional.

| Rol | Nombre | Firma | Fecha |
|---|---|---|---|
| Responsable del proyecto |  |  |  |
| Responsable de datos |  |  |  |
| Responsable técnico |  |  |  |
| Docente/asesor, si aplica |  |  |  |

> Este documento no se considera firmado hasta completar al menos las firmas internas exigidas por el equipo.

## Notas de implementación de Plan 0

- **[INGENIERÍA] Entorno aislado:** Ubuntu 26.04 incluye Python 3.14.4 y no ofrece en la auditoría un intérprete 3.10 alternativo. Se instaló CPython 3.10.18 mediante `uv==0.7.13` y se creó `.venv` sin reemplazar el Python del sistema.
- **[INGENIERÍA] Validación de versión:** el test compara `sys.version_info[:2]` con el intervalo 3.7–3.10. Comparar el triplete contra `(3, 10, 0)` rechazaría incorrectamente parches compatibles como 3.10.18, aunque utilicen el wheel ABI `cp310` publicado para Procgen 0.10.7.
- **[INGENIERÍA] Transición de smoke test:** `ProcgenGym3Env` expone la API Gym3 `act()`/`observe()`, no `step()`/`reset()`. El smoke test ejecuta exactamente un `act()` determinista y ningún otro test avanza el simulador.
