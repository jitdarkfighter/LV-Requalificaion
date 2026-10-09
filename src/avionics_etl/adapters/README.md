# Writing a source adapter

The pipeline is deliberately built so that **adding a new data source touches
exactly two files**: one adapter here, and one channel registry in `configs/`.
No transform, test, warehouse or dashboard code changes.

This matters because C-MAPSS is a stand-in. The real telemetry schema from ISRO
has not been shared with us, and when it arrives it will have different
identifiers, different channel names, different units and a different notion of
"cycle". See `docs/PRIMER.md` §10.

---

## What an adapter must do

Subclass `SourceAdapter` (`base.py`) and implement three methods.

### 1. `discover() -> list[Batch]`

List what is available to ingest. A `Batch` is one **arrival unit** — for
C-MAPSS, one source file; for real data, most likely one recovery event or one
telemetry dump.

```python
Batch(
    batch_id="TV-D1-flight-07",       # unique, stable, human-readable
    source_id="TV-D1",                 # which source/campaign
    mission_id="TV-D1-07",             # which mission
    files=(path_to_data,),             # everything the batch needs
    trajectory_complete=False,         # did this asset run to failure?
    metadata={...},                    # anything source-specific
)
```

`batch_id` must be **stable across runs**. It is the watermark key: ingest skips
batches already recorded, and re-ingests only if their file digests changed.

`trajectory_complete` is the canonical form of C-MAPSS's train/test split. A
recovered avionics box has *not* failed, so its trajectory is incomplete —
which is exactly the `test` case. It decides which labelling path applies.

### 2. `read_chunks(batch, *, chunk_rows) -> Iterator[DataFrame]`

Yield the batch as canonical frames of at most `chunk_rows` rows.

**This must be a generator.** Peak memory has to stay proportional to the chunk
size, not the batch size — that is what makes a 64 GB arrival ingestible on one
machine. Never read a whole batch into memory.

Every yielded frame must carry the canonical entity columns:

| Column | Meaning |
|---|---|
| `source_id` | which data source |
| `mission_id` | which mission or campaign |
| `asset_id` | the physical thing being tracked (unique within source+mission) |
| `sequence_no` | monotonic usage counter, starting at 1 |

plus `batch_id`, `trajectory_complete`, and one column per registered channel.

`model.assert_canonical(df.columns)` will tell you if you have missed one.

> **Emit chunks already sorted by `(asset_id, sequence_no)`.** The writer appends
> row groups in arrival order, and sorted data is what keeps Parquet row-group
> statistics tight enough for predicate pushdown to work. Ingest verifies this
> after writing and warns if it was not honoured. If your source cannot emit
> sorted chunks, it needs an external sort pass before ingest.

### 3. `registry() -> ChannelRegistry`

Return the channels this source provides, loaded from a YAML registry. Copy
`configs/channels.yaml` as a starting point and declare, per channel:

```yaml
channels:
  rail_voltage_5v:
    role: sensor          # sensor | setting | discrete | derived
    symbol: V5
    quantity: 5V rail voltage
    unit: volt
    dtype: float32
    bounds: [4.75, 5.25]  # physically plausible; violations are quarantined
    max_rate: 0.10        # largest legal change between consecutive steps
```

`role` matters:

- **`sensor`** / **`discrete`** — measure the *asset's* state. Candidate features,
  normalised within regime.
- **`setting`** — describe the *operating condition*. These define the regime;
  they are not normalised within it. See `docs/PRIMER.md` §6.

### Optional: `read_external_labels(batch) -> DataFrame | None`

When ground truth arrives separately from the telemetry (as C-MAPSS's
`RUL_FD00x.txt` does), return it here. The real system will be in this position
too: a recovered box has not failed, so its remaining life cannot come from its
own telemetry.

---

## Then register it

One entry in `extract/ingest.py`:

```python
def build_adapter(cfg, name):
    if name == "cmapss":
        return CmapssAdapter(...)
    if name == "isro":
        return IsroAdapter(...)      # <- your new adapter
```

and set `ingest.adapter` in `configs/pipeline.yaml`.

---

## Check your work

```bash
python -m avionics_etl.extract.ingest --adapter <name> --list   # discovery
python -m avionics_etl.extract.ingest --adapter <name>          # ingest
make audit                                                      # integrity
make test                                                       # leakage invariants
```

`scripts/audit_dataset.py` is written against the canonical model, so most of it
applies to any source. The C-MAPSS-specific expectations are isolated in the
constants at the top of that file — replace those, keep the rest.

---

## What NOT to put in an adapter

An adapter **reshapes**. It does not clean, filter, normalise or label.

| Belongs in the adapter | Belongs downstream |
|---|---|
| Parsing the file format | Bound and rate checks (`transform/validate.py`) |
| Renaming columns to canonical names | Outlier flagging (`transform/outliers.py`) |
| Unit conversion into the declared unit | Regime assignment (`transform/regime.py`) |
| Assembling entity columns | Normalisation (`transform/normalize.py`) |
| Chunking for bounded memory | Labels, features, splits |

If you find yourself writing a threshold in an adapter, it belongs in the
channel registry instead.
