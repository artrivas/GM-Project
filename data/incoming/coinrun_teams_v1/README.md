# CoinRun teams v1 — entrega externa aislada

Este dataset fue recibido originalmente en la carpeta raíz `CoinRun/` y se movió sin convertir a esta zona de ingreso. **No forma parte de `data/raw`, `data/splits` ni de los entrenamientos activos.**

## Estructura

```text
coinrun_teams_v1/
├── original/                 # entrega original intacta por Team/session
├── index/
│   ├── inventory.json        # auditoría cuantitativa
│   ├── manifest.jsonl        # un registro normalizado por NPZ válido
│   ├── quarantine.jsonl      # referencias ausentes o archivos inválidos
│   └── source_files.sha256   # integridad de la entrega original
├── DO_NOT_INGEST.json        # bloqueo explícito
└── README.md
```

## Inventario verificado

- 6 equipos y 21 sesiones.
- 1,899 NPZ válidos y 163,746 frames RGB `uint8` 64×64.
- 859 episodios `easy` y 1,040 `hard`.
- 1,421 victorias, 468 terminales no ganadores y 10 truncamientos.
- 1,899 seeds distintos y ningún grupo de contenido duplicado.
- Cinco archivos declarados por Team5/session_20260524_125456 no fueron entregados; están registrados en `quarantine.jsonl`.

## Por qué no se incorpora automáticamente

1. No existe `source_policy`; `Team*` no se interpreta automáticamente como gameplay humano.
2. Faltan `initial_state`, `next_observations`, `dones` y metadata por transición exigida por el esquema activo.
3. `num_levels=0` y los 1,899 seeds procedurales no corresponden al alcance temprano de seed fijo.
4. Los no ganadores no permiten distinguir muerte de caída.
5. Los splits existentes fueron auditados y no deben regenerarse implícitamente.

## Repetir el inventario

Desde la raíz del repositorio:

```bash
source .venv/bin/activate
python -m src.data_prep.inventory_external_coinrun \
  --root data/incoming/coinrun_teams_v1/original \
  --out data/incoming/coinrun_teams_v1/index
```

La integración futura requerirá una fase explícita de adaptación, auditoría y regeneración de splits. Nunca copiar estos NPZ directamente a `data/raw/`.
