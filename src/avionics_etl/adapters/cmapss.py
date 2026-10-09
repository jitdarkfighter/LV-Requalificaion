"""C-MAPSS source adapter.

The only module in the pipeline that knows C-MAPSS's file layout and column
names. Everything else works in the canonical model.

Source shape: eight headerless, space-separated text files, 26 columns each —
unit number, cycle, three operational settings, twenty-one sensors. See
docs/PRIMER.md section 5.

Mapping to canonical form::

    unit number  -> asset_id   (natural key — only unique within one file)
                 -> asset_uid  ("{batch}-{unit:04d}" — the real identity)
    cycle        -> sequence_no
    FD00x        -> source_id  (and mission_id — C-MAPSS has no mission concept,
                                so each subset is treated as one simulation
                                campaign)
    train/test   -> trajectory_complete  (train runs to failure; test is
                                          truncated, exactly like a recovered
                                          avionics box that has not failed)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pandas as pd

from avionics_etl.adapters.base import DEFAULT_CHUNK_ROWS, Batch, SourceAdapter
from avionics_etl.canonical import model
from avionics_etl.canonical.model import ChannelRegistry

N_SENSORS = 21
N_OP_SETTINGS = 3

#: Column order in the source files. Headerless, so this is positional.
SOURCE_COLUMNS = (
    ["unit_number", "cycle"]
    + [f"op_setting_{i}" for i in range(1, N_OP_SETTINGS + 1)]
    + [f"s{i}" for i in range(1, N_SENSORS + 1)]
)

RENAME = {"unit_number": model.ASSET_ID, "cycle": model.SEQUENCE_NO}

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

#: C-MAPSS split -> whether the trajectory runs all the way to failure.
SPLIT_COMPLETE = {"train": True, "test": False}


class CmapssAdapter(SourceAdapter):
    """Reads NASA C-MAPSS (``CMAPSSData.zip``) into canonical frames."""

    name = "cmapss"

    def __init__(self, root: Path, registry_path: Path, subsets: list[str] | None = None):
        self.root = Path(root)
        self.registry_path = Path(registry_path)
        self.subsets = list(subsets) if subsets else list(SUBSETS)

    # -- discovery ----------------------------------------------------------

    def discover(self) -> list[Batch]:
        """One batch per source file — eight in total.

        Treating each file as a separate arrival is not artificial bookkeeping:
        it exercises the incremental ingest path, so the same code that loads
        C-MAPSS today loads mission-by-mission arrivals later.
        """
        batches: list[Batch] = []
        for subset in self.subsets:
            for split, complete in SPLIT_COMPLETE.items():
                path = self.root / f"{split}_{subset}.txt"
                if not path.exists():
                    continue
                extras: list[Path] = []
                if not complete:
                    # Truncated trajectories ship their true RUL separately.
                    rul = self.root / f"RUL_{subset}.txt"
                    if rul.exists():
                        extras.append(rul)
                batches.append(
                    Batch(
                        batch_id=f"{subset}-{split}",
                        source_id=subset,
                        mission_id=subset,
                        files=(path, *extras),
                        trajectory_complete=complete,
                        metadata={"cmapss_split": split, "subset": subset},
                    )
                )
        return batches

    # -- reading ------------------------------------------------------------

    def read_chunks(
        self, batch: Batch, *, chunk_rows: int = DEFAULT_CHUNK_ROWS
    ) -> Iterator[pd.DataFrame]:
        measurement_file = batch.files[0]
        registry = self.registry()
        dtypes = {
            "unit_number": "int32",
            "cycle": "int32",
            **{name: registry[name].dtype for name in registry.names()},
        }

        reader = pd.read_csv(
            measurement_file,
            sep=r"\s+",
            header=None,
            names=SOURCE_COLUMNS,
            dtype=dtypes,
            chunksize=chunk_rows,
        )
        for chunk in reader:
            yield self._to_canonical(chunk, batch, measurement_file.name)

    def _to_canonical(self, chunk: pd.DataFrame, batch: Batch, filename: str) -> pd.DataFrame:
        if list(chunk.columns) != SOURCE_COLUMNS:
            raise ValueError(f"{filename}: unexpected columns {list(chunk.columns)}")
        if chunk.isna().any().any():
            bad = chunk.columns[chunk.isna().any()].tolist()
            raise ValueError(f"{filename}: unparseable values in {bad}")

        df = chunk.rename(columns=RENAME)
        # Unit numbers restart at 1 in every file, so the natural key is not a
        # global identity: FD001-train unit 1 and FD001-test unit 1 are
        # different engines. Namespace by batch to get a real one.
        df.insert(0, model.ASSET_UID,
                  batch.batch_id + "-" + df[model.ASSET_ID].astype(str).str.zfill(4))
        df.insert(0, model.TRAJECTORY_COMPLETE, batch.trajectory_complete)
        df.insert(0, model.BATCH_ID, batch.batch_id)
        df.insert(0, model.MISSION_ID, batch.mission_id)
        df.insert(0, model.SOURCE_ID, batch.source_id)

        model.assert_canonical(df.columns)
        return df

    def read_external_labels(self, batch: Batch) -> pd.DataFrame | None:
        """True RUL for truncated trajectories, from ``RUL_FD00x.txt``.

        Only incomplete trajectories have these: a complete trajectory's label is
        derived from its own end. The real system will be in exactly this
        position — a recovered box has not failed, so its remaining life can only
        come from outside the telemetry.
        """
        if batch.trajectory_complete or len(batch.files) < 2:
            return None
        values = [int(v) for v in batch.files[1].read_text().split()]
        return pd.DataFrame(
            {
                model.SOURCE_ID: batch.source_id,
                model.MISSION_ID: batch.mission_id,
                model.BATCH_ID: batch.batch_id,
                model.ASSET_ID: pd.Series(range(1, len(values) + 1), dtype="int32"),
                model.ASSET_UID: [f"{batch.batch_id}-{i:04d}" for i in range(1, len(values) + 1)],
                "rul_true": pd.Series(values, dtype="int32"),
            }
        )

    # -- registry -----------------------------------------------------------

    def registry(self) -> ChannelRegistry:
        return ChannelRegistry.from_yaml(self.registry_path)
