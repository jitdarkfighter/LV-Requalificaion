"""Generate synthetic launch-vehicle checkout telemetry.

Why synthetic
-------------
The real data is restricted, and the reference study this work follows took the
same route for the same reason: it generated a synthetic dataset "with
characteristics similar to launch vehicle test data" rather than publishing
mission data.

Synthetic data is only useful if the structure a model is supposed to find is
actually present. Random noise would be worthless. So the generator reproduces
the three properties that make real checkout data hard:

**Event-driven, not periodic.** A checkout computer issues a sequence of
commands and parameters respond to them. Nothing here is a sine wave; the
signals are piecewise, with transitions at command times. That is exactly why
periodicity-based anomaly detection fails on this data.

**Spatial relationships between parameters.** The reference study is specific
about these, and they are built in rather than approximated:

* current and terminal voltage are *linearly* related — switching a load on
  pulls the bus voltage down by ``I x R``
* current and heat flux are *higher order* — resistive heating goes as ``I^2``
* temperature *lags* heat flux, being its thermal integral
* package status and current monitoring are *directly linked* — the status bit
  going high is what causes the current to appear

A model that only looks at one channel at a time cannot detect a fault in those
couplings, which is the whole argument for multivariate modelling.

**Redundancy.** Flight hardware carries backup instrumentation. The ``_1``/``_2``
suffixed pairs are genuine redundant pairs: same physical quantity, independent
sensors, slightly different noise and calibration offset. A disagreement between
them is a fault that neither reading reveals on its own.

Parameter names follow the reference study's twelve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import pandas as pd

Kind = Literal["static", "dynamic", "discrete"]

#: Channels, in the reference study's order. ``redundant_with`` names the prime
#: channel a backup shadows.
@dataclass(frozen=True)
class ChannelSpec:
    """One channel's physical description.

    ``expected_kind`` records what this channel does *under the default test
    profile*. It is documentation, not an input: classification is always
    data-driven, because which parameters a test affects depends on the test.
    A profile that never opens a valve leaves ``Pressure_1`` genuinely static.
    """

    name: str
    expected_kind: Kind
    unit: str
    ambient: float
    bounds: tuple[float, float]
    noise: float
    redundant_with: str | None = None
    offset: float = 0.0          # calibration offset of a backup sensor


CHANNELS: tuple[ChannelSpec, ...] = (
    ChannelSpec("RMS_Current_1",      "dynamic",  "A",      0.0,   (-1.0, 40.0),   0.015),
    ChannelSpec("RMS_Current_2",      "dynamic",  "A",      0.0,   (-1.0, 40.0),   0.018, "RMS_Current_1", 0.03),
    ChannelSpec("Battery_Voltage",    "dynamic",  "V",      28.0,  (20.0, 32.0),   0.008),
    ChannelSpec("Skin_Temperature_1", "dynamic",  "degC",   22.0,  (-20.0, 90.0),  0.012),
    ChannelSpec("Skin_Temperature_2", "dynamic",  "degC",   22.0,  (-20.0, 90.0),  0.014, "Skin_Temperature_1", 0.08),
    ChannelSpec("Force_1",            "dynamic",  "N",      0.0,   (-50.0, 600.0), 0.35),
    ChannelSpec("Force_2",            "dynamic",  "N",      0.0,   (-50.0, 600.0), 0.40, "Force_1", 0.9),
    ChannelSpec("Heat_Flux_1",        "dynamic",  "W_m2",   0.0,   (-10.0, 900.0), 0.60),
    ChannelSpec("Heat_Flux_2",        "dynamic",  "W_m2",   0.0,   (-10.0, 900.0), 0.65, "Heat_Flux_1", 1.2),
    ChannelSpec("Temperature_1",      "dynamic",  "degC",   25.0,  (-20.0, 150.0), 0.030),
    ChannelSpec("Temperature_2",      "dynamic",  "degC",   25.0,  (-20.0, 150.0), 0.034, "Temperature_1", 0.12),
    ChannelSpec("Pressure_1",         "dynamic",  "bar",    1.013, (0.0, 12.0),    0.0012),
)

CHANNEL_NAMES = tuple(c.name for c in CHANNELS)

REDUNDANT_PAIRS = tuple(
    (c.redundant_with, c.name) for c in CHANNELS if c.redundant_with is not None
)

#: Bus resistance: how far the rail sags per amp drawn. Linear, per the study.
BUS_RESISTANCE = 0.055

#: Resistive heating coefficient — heat flux rises with the SQUARE of current.
HEATING_COEFF = 0.75

#: Thermal time constant, in samples, for temperature lagging heat flux.
THERMAL_TAU = 220.0


@dataclass(frozen=True)
class Command:
    """One checkout command and the response it provokes."""

    at: int                     # sample index the command is issued
    name: str
    load_amps: float = 0.0      # current the switched package draws
    force_newtons: float = 0.0  # actuator demand
    pressure_bar: float = 0.0   # valve effect
    rise: int = 40              # samples the response takes to settle


@dataclass
class TestProfile:
    """A checkout test: how long it runs and what commands it issues."""

    name: str
    n_samples: int = 9000
    rate_hz: float = 61.0
    commands: Sequence[Command] = field(default_factory=tuple)


def default_profile(name: str = "checkout-01", n_samples: int = 9000) -> TestProfile:
    """A representative checkout sequence.

    Commands are spread across the run so most channels spend most of the time
    at ambient — which is the real situation, and the reason variance-based
    parameter classification is worth doing at all.
    """
    n = n_samples
    return TestProfile(
        name=name,
        n_samples=n,
        commands=(
            Command(int(n * 0.08), "switch_to_battery",   load_amps=6.5),
            Command(int(n * 0.18), "avionics_bus_on",     load_amps=11.0),
            Command(int(n * 0.30), "actuator_slew_a",     force_newtons=180.0, rise=90),
            Command(int(n * 0.42), "valve_open",          pressure_bar=4.2,    rise=60),
            Command(int(n * 0.52), "actuator_slew_b",     force_newtons=420.0, rise=120),
            Command(int(n * 0.66), "heater_on",           load_amps=9.0),
            Command(int(n * 0.78), "actuator_return",     force_newtons=40.0,  rise=90),
            Command(int(n * 0.88), "valve_close",         pressure_bar=0.0,    rise=60),
        ),
    )


def power_only_profile(name: str = "power-check", n_samples: int = 9000) -> TestProfile:
    """A narrower test that only exercises the power chain.

    Force and pressure are never commanded, so those channels sit at ambient for
    the whole run. This is the normal case the reference study describes — a
    given test affects a small subset of parameters and the rest must be
    verified to have stayed put — and it is what gives the static-parameter
    limit check something real to check.
    """
    n = n_samples
    return TestProfile(
        name=name,
        n_samples=n,
        commands=(
            Command(int(n * 0.12), "switch_to_battery", load_amps=6.5),
            Command(int(n * 0.35), "avionics_bus_on",   load_amps=11.0),
            Command(int(n * 0.70), "heater_on",         load_amps=9.0),
        ),
    )


def _ramp(n: int, at: int, rise: int, amplitude: float) -> np.ndarray:
    """A command response: flat, then a smooth transition, then held.

    Uses a raised-cosine rather than a step so the derivative is continuous.
    Real actuators and power supplies do not change instantaneously, and a true
    step would make the rate-of-change contract checks fire on normal operation.
    """
    out = np.zeros(n)
    if amplitude == 0.0:
        return out
    rise = max(int(rise), 1)
    end = min(at + rise, n)
    if at < n:
        k = np.arange(end - at)
        out[at:end] = amplitude * 0.5 * (1 - np.cos(np.pi * k / max(rise, 1)))
        out[end:] = amplitude
    return out


def _lag(x: np.ndarray, tau: float) -> np.ndarray:
    """First-order lag — the discrete form of a thermal mass heating up."""
    alpha = 1.0 / max(tau, 1.0)
    out = np.empty_like(x)
    acc = 0.0
    for i, v in enumerate(x):
        acc += alpha * (v - acc)
        out[i] = acc
    return out


def generate(
    profile: TestProfile | None = None,
    *,
    seed: int = 0,
    mission_id: str = "LV-M01",
    asset_id: str = "AV-001",
    start_time: float = 0.0,
) -> pd.DataFrame:
    """Generate one checkout test as a tidy frame.

    Returns columns: ``time`` (seconds, absolute), ``mission_id``, ``asset_id``,
    ``test_name``, plus one column per channel.
    """
    profile = profile or default_profile()
    rng = np.random.default_rng(seed)
    n = profile.n_samples
    t = start_time + np.arange(n) / profile.rate_hz

    # --- command responses -------------------------------------------------
    load = np.zeros(n)
    force_cmd = np.zeros(n)
    pressure = np.full(n, 1.013)
    for cmd in profile.commands:
        load += _ramp(n, cmd.at, cmd.rise, cmd.load_amps)

        # Each command ramps a quantity from wherever it currently sits to the
        # new demand. Ramping from zero instead would make the signal jump
        # discontinuously at every command after the first.
        if cmd.force_newtons:
            current_level = force_cmd[max(cmd.at - 1, 0)]
            force_cmd = force_cmd + _ramp(n, cmd.at, cmd.rise, cmd.force_newtons - current_level)

        if cmd.pressure_bar or cmd.name.endswith("close"):
            target = cmd.pressure_bar if cmd.pressure_bar else 1.013
            step = _ramp(n, cmd.at, cmd.rise, target - pressure[max(cmd.at - 1, 0)])
            pressure = pressure + step

    # --- couplings ---------------------------------------------------------
    current = load                                            # status -> current
    voltage = 28.0 - BUS_RESISTANCE * current                 # linear
    heat_flux = HEATING_COEFF * current**2                    # higher order
    temperature = 25.0 + _lag(heat_flux, THERMAL_TAU) * 0.08  # lagged integral
    skin = 22.0 + _lag(heat_flux, THERMAL_TAU * 6) * 0.010    # much slower

    base = {
        "RMS_Current_1": current,
        "Battery_Voltage": voltage,
        "Force_1": force_cmd,
        "Heat_Flux_1": heat_flux,
        "Temperature_1": temperature,
        "Skin_Temperature_1": skin,
        "Pressure_1": pressure,
    }

    data: dict[str, np.ndarray] = {}
    for spec in CHANNELS:
        if spec.redundant_with is not None:
            signal = base[spec.redundant_with] + spec.offset
        else:
            signal = base[spec.name]
        data[spec.name] = signal + rng.normal(0.0, spec.noise, n)

    return pd.DataFrame(
        {
            "time": t,
            "mission_id": mission_id,
            "asset_id": asset_id,
            "test_name": profile.name,
            **data,
        }
    )


def generate_missions(
    n_missions: int = 3,
    *,
    n_samples: int = 9000,
    seed: int = 0,
    stagger_seconds: float = 86_400.0,
) -> pd.DataFrame:
    """Several missions, each stamped with its own absolute start time.

    The staggered timestamps are deliberate: they are what makes mission
    synchronisation necessary, and a generator that emitted everything starting
    at zero would quietly hide that requirement.
    """
    frames = []
    for i in range(n_missions):
        frames.append(
            generate(
                default_profile(f"checkout-{i + 1:02d}", n_samples),
                seed=seed + i,
                mission_id=f"LV-M{i + 1:02d}",
                asset_id=f"AV-{i + 1:03d}",
                start_time=i * stagger_seconds,
            )
        )
    return pd.concat(frames, ignore_index=True)


def redundancy_pairs(*, sigma: float = 6.0):
    """Declared redundant pairs with a tolerance for each.

    A single flat tolerance cannot work: two sensors measuring a 900 W/m^2 heat
    flux will disagree by more than two measuring a 28 V rail, purely from
    calibration offset and noise, with nothing wrong. In the real system these
    tolerances come from the measurement plan; here they are derived from each
    pair's known calibration offset plus a margin on the combined sensor noise.

    Importing here rather than at module scope keeps the generator usable
    without the transform package.
    """
    from avionics_etl.transform.classify import RedundancyPair

    by_name = {c.name: c for c in CHANNELS}
    out = []
    for prime, backup in REDUNDANT_PAIRS:
        b = by_name[backup]
        combined_noise = (by_name[prime].noise**2 + b.noise**2) ** 0.5
        out.append(
            RedundancyPair(
                prime=prime,
                backup=backup,
                tolerance=abs(b.offset) + sigma * combined_noise,
                unit=b.unit,
            )
        )
    return out


def registry_yaml() -> str:
    """Channel registry for this source, in the pipeline's declared format."""
    lines = [
        "# Channel registry for synthetic launch-vehicle checkout telemetry.",
        "# Generated by avionics_etl.synthetic.lv_telemetry.registry_yaml().",
        "",
        "version: 1",
        "source: lv_synthetic",
        "",
        "channels:",
    ]
    for spec in CHANNELS:
        role = "discrete" if spec.expected_kind == "discrete" else "sensor"
        lines += [
            f"  {spec.name}:",
            f"    role: {role}",
            f"    symbol: {spec.name}",
            f"    quantity: {spec.name.replace('_', ' ')}",
            f"    unit: {spec.unit}",
            "    dtype: float32",
            f"    bounds: [{spec.bounds[0]}, {spec.bounds[1]}]",
            f"    max_rate: {max(abs(spec.bounds[1] - spec.bounds[0]) * 0.25, 1.0):.4f}",
            f"    observed: [{spec.bounds[0]}, {spec.bounds[1]}]",
        ]
    return "\n".join(lines) + "\n"
