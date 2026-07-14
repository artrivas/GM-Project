# Auditoría técnica independiente del dataset — Plan 2

## Veredicto

**GO CONDICIONAL para iniciar Plan 3 como entrenamiento preliminar del autoencoder.**

Las cifras elegibles recalculadas coinciden con `quality_report.json`, existen ejemplos globales de victoria y falla terminal, y no hay fuga entre splits. La condición es estricta: este dataset sirve para verificar el pipeline y estudiar reconstrucción visual sobre el seed 0, pero no constituye evidencia de cobertura suficiente. Plan 5 queda bloqueado hasta corregir la ausencia total de eventos raros en `val` y `test`.

El `NO-GO` emitido por la implementación anterior usa puertas más estrictas (tres eventos únicos y gameplay humano). No es una discrepancia de datos. Aplicando literalmente los criterios K de este prompt de auditoría, corresponde `GO CONDICIONAL`, porque los eventos existen globalmente pero su cobertura por split es baja.

## Método independiente

La revisión abrió directamente los HDF5 con `h5py` y los manifests como JSONL. No importó `audit_dataset.py`, `common.py` ni `split_dataset.py`. Para cada episodio se contrastaron atributos (`win`, `death`, `truncated`) contra el último código de `events` y los últimos valores de `dones`/`truncateds`. Para la fuga se reabrió cada ruta declarada por los manifests y se recalcularon IDs y SHA-256 del contenido de transición.

Evidencia reproducible:

- `independent_recount.json`: recálculo directo de `data/raw/` y comparación con el reporte.
- `independent_leakage_check.json`: recuento por split y solapamientos.
- `independent_test_output.log`: ejecución del test de fuga solicitado.

## 1. Frames y episodios reales

| Universo | Episodios | Frames | Victorias | Fallas/deaths |
|---|---:|---:|---:|---:|
| Todos los HDF5 físicamente presentes | 23 | 5,460 | 5 | 6 |
| Excluidos de entrenamiento | 3 | 1,775 | 0 | 1 |
| Elegibles | 20 | 3,685 | 5 | 5 |
| Afirmados por `quality_report.json` | 20 | 3,685 | 5 | 5 |

Los excluidos son dos episodios `gui_smoke_idle` de 1,000 y 755 frames, y un `smoke_random` de 20 frames. Por tanto, 5,460 es el total físico bajo `data/raw/`, mientras que 3,685 es el total realmente asignable al dataset. La cifra del reporte coincide con su alcance elegible.

**[HALLAZGO]** No hay discrepancias en episodios, frames, victorias, fallas ni distribución por seed para el universo elegible. Tampoco se encontraron inconsistencias entre los atributos terminales y el código final de evento.

## 2. Muertes y victorias por split

Los conteos siguientes provienen de reabrir los HDF5 apuntados por cada manifest, no de `summary.json`:

| Split | Episodios | Frames | Victorias físicas/únicas | Fallas físicas/únicas | Truncados |
|---|---:|---:|---:|---:|---:|
| train | 16 | 2,485 | 5 / 1 | 5 / 1 | 6 |
| val | 2 | 600 | 0 / 0 | 0 / 0 | 2 |
| test | 2 | 600 | 0 / 0 | 0 / 0 | 2 |

**[HALLAZGO]** Las cinco victorias de train son copias de una sola trayectoria de contenido; lo mismo ocurre con las cinco fallas. Val y test no contienen ningún evento raro. El dataset cumple la existencia global mínima del criterio K, pero no permite evaluar reconstrucción específica de victorias o fallas fuera de train y no es una prueba significativa para Plan 5.

## 3. Representación de seeds

| Seed | Episodios elegibles | Porcentaje |
|---:|---:|---:|
| 0 | 20 | 100% |

**[HALLAZGO]** No puede medirse sobrerrepresentación relativa entre seeds porque solo existe uno; operacionalmente, seed 0 concentra el 100% de los datos. Esto es compatible con el alcance temprano de seed fijo, pero impide cualquier conclusión de generalización y hace que train/val/test compartan el mismo seed.

## 4. Episodios truncados sin `done`

Existen **10 episodios truncados con `done=False`** en la última transición. Los 10 tienen simultáneamente `truncated=True`, atributo `truncated=1` y código final de evento 3.

**[HALLAZGO]** Son truncamientos sin `done` explícito, pero no están silenciosamente sin marcar: el esquema representa terminación del entorno y truncamiento de horizonte como señales separadas. Un consumidor posterior debe usar `done OR truncated` como límite de secuencia; usar solo `done` concatenaría incorrectamente estos diez episodios.

## 5. Fuga entre splits

- Solapamiento de `episode_id` train/val, train/test y val/test: **0**.
- Solapamiento de huellas de contenido entre cualquier par: **0**.
- IDs elegibles sin asignar: **0**.
- IDs asignados que no sean elegibles: **0**.
- Frames reabiertos desde manifests: 2,485 + 600 + 600 = **3,685**.

**[HALLAZGO]** La división evita fuga por ID y también mantiene juntas las trayectorias byte-equivalentes. Esta verificación coincide con el reporte original.

## Condiciones antes de Plan 5

1. Recolectar al menos **2 trayectorias adicionales y distintas de victoria** y **2 de falla terminal** en seed 0, para alcanzar un mínimo global de 3 únicas de cada clase y poder asignar una a cada split.
2. Para victorias, partir del modo `hard` que produjo los éxitos actuales, introduciendo variación real de timing/acciones para no repetir la misma huella.
3. Para fallas, partir del modo `easy` que produjo las fallas actuales, también con trayectorias de acciones distintas.
4. Añadir una sesión humana válida con acciones no-noop, reauditar, regenerar el split y exigir al menos una victoria y una falla únicas en train, val y test.
5. En cualquier cargador secuencial, cortar episodios con `done OR truncated`.

Hasta cumplir estas condiciones, Plan 3 solo debe considerarse una prueba preliminar de reconstrucción sobre datos limitados, no una validación del dataset definitivo.
