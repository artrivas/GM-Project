# Resultados de EXP-002

## Auditoría

- 23 archivos HDF5 inspeccionados: 20 elegibles, 3 excluidos y 0 rotos.
- 3,685 frames elegibles y cobertura de las 15 acciones Procgen.
- 5 victorias y 5 fallas terminales físicas, pero solo 1 victoria y 1 falla únicas después de agrupar trayectorias idénticas.
- 12 trayectorias de contenido únicas; 10 episodios pertenecen a 2 grupos duplicados.
- No existe una sesión humana válida; las dos capturas inactivas permanecen excluidas.
- La muestra visual aleatoria de dos episodios no mostró frames vacíos, corrupción, mezcla de episodios ni desaparición anómala del jugador.

## Split diagnóstico

| Split | Episodios | Frames | Victorias físicas/únicas | Fallas físicas/únicas |
|---|---:|---:|---:|---:|
| train | 16 | 2,485 | 5 / 1 | 5 / 1 |
| val | 2 | 600 | 0 / 0 | 0 / 0 |
| test | 2 | 600 | 0 / 0 | 0 / 0 |

Los 20 IDs, los 3,685 frames y las 12 huellas se conservan exactamente una vez. No hay solapamiento de IDs ni huellas entre splits.

## Decisión

**No-Go para Plan 3.** Val y test no contienen ninguna victoria ni falla terminal única, y Plan 1 no produjo gameplay humano válido. Los manifests existen para evidenciar el algoritmo, pero están marcados `DO_NOT_USE`. Para reabrir el Go se necesitan al menos tres trayectorias de victoria únicas, tres de falla únicas y una sesión humana real, seguidas de una nueva auditoría.
