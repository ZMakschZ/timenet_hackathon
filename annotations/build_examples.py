"""Build the Hill of Towie annotation pilot from Zenodo 22662930 annual ZIPs."""

import argparse
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scenario_answers import ANSWERS

CHANNELS = {
    "P": ("Grid", "wtc_ActPower_mean", "kW"),
    "W": ("Turbine", "wtc_AcWindSp_mean", "m/s"),
    "R": ("Turbine", "wtc_GenRpm_mean", "rpm"),
    "C": ("Turbine", "wtc_PowerRef_endvalue", "kW"),
    "S": ("Turbine", "wtc_MainSRpm_mean", "rpm"),
    "ON": ("Flag", "wtc_ScInOper_timeon", "s"),
    "READY": ("Flag", "wtc_ScReToOp_timeon", "s"),
    "ERR": ("Flag", "wtc_ScTurSto_timeon", "s"),
    "B": ("Turbine", "wtc_PitcPosA_mean", "deg"),
    "A": ("Temp", "wtc_AmbieTmp_mean", "°C"),
    "U": ("Temp", "wtc_Gen1U1Tm_mean", "°C"),
    "V": ("Temp", "wtc_Gen1V1Tm_mean", "°C"),
    "Z": ("Temp", "wtc_Gen1W1Tm_mean", "°C"),
    "VR": ("Grid", "wtc_VoltPhR_mean", "V"),
    "VS": ("Grid", "wtc_VoltPhS_mean", "V"),
    "VT": ("Grid", "wtc_VoltPhT_mean", "V"),
    "IR": ("Grid", "wtc_AmpPhR_mean", "A"),
    "IS": ("Grid", "wtc_AmpPhS_mean", "A"),
    "IT": ("Grid", "wtc_AmpPhT_mean", "A"),
    "Q": ("Grid", "wtc_ReactPwr_mean", "kvar"),
}
TABLES = {
    "Turbine": "tblSCTurbine",
    "Grid": "tblSCTurGrid",
    "Temp": "tblSCTurTemp",
    "Flag": "tblSCTurFlag",
}
GROUPS = {
    "temp": ["U", "V", "Z"],
    "volt": ["VR", "VS", "VT"],
    "amp": ["IR", "IS", "IT"],
}
SPLITS = {
    "train": ["2024_07", "2024_08", "2024_09"],
    "validation": ["2024_10", "2024_11", "2024_12"],
    "stress_test": ["2026_03"],
}
RATED = 2300
RECORD = "https://zenodo.org/records/22662930"


def read_month(directory, month, *, stations=None):
    first = pd.Timestamp(month.replace("_", "-") + "-01", tz="UTC")
    grid = pd.date_range(
        first, first + pd.offsets.MonthBegin(), freq="10min", inclusive="right"
    )
    frames = []
    with ZipFile(directory / f"{month[:4]}.zip") as archive:
        for table, prefix in TABLES.items():
            names = {
                field: short
                for short, (group, field, _) in CHANNELS.items()
                if group == table
            }
            with archive.open(f"{prefix}_{month}.csv") as stream:
                frame = pd.read_csv(stream, usecols=["StationId", "TimeStamp", *names])
            frame["TimeStamp"] = pd.to_datetime(
                frame["TimeStamp"], format="ISO8601", utc=True
            )
            frame = frame.rename(columns=names).set_index(["StationId", "TimeStamp"])
            frame = frame.loc[frame.index.get_level_values("TimeStamp").isin(grid)]
            if frame.index.has_duplicates:
                raise ValueError(f"Duplicate raw keys in {month} {prefix}")
            frames.append(frame)
    data = pd.concat(frames, axis=1).apply(pd.to_numeric, errors="raise")
    data = data.replace([np.inf, -np.inf], np.nan)
    observed = {
        int(station): frame.droplevel(0).reindex(grid).round(3)
        for station, frame in data.groupby(level=0)
    }
    if stations is None:
        return observed
    if unknown := set(observed) - set(stations):
        raise ValueError(f"{month} has unknown station IDs {sorted(unknown)}")
    return {
        station: observed.get(
            station, pd.DataFrame(np.nan, index=grid, columns=list(CHANNELS))
        )
        for station in stations
    }


def spreads(frame, fields):
    return frame[fields].max(axis=1, skipna=False) - frame[fields].min(
        axis=1, skipna=False
    )


def select(frame):
    p, _w, r, c, on, _ready, err, _b = (
        frame[key] for key in ("P", "W", "R", "C", "ON", "READY", "ERR", "B")
    )
    last = frame.rolling(7)
    lo, hi = last.min(), last.max()
    band = hi - lo
    change = frame.diff(6)
    run = lo.ON.eq(600)
    shortfall = c - p
    t = spreads(frame, GROUPS["temp"])
    v = spreads(frame, GROUPS["volt"])
    i = spreads(frame, GROUPS["amp"])
    near_flat_w = band.W.le(0.25)
    missing = frame.isna().all(axis=1)
    temperature = frame[["A", *GROUPS["temp"]]]
    temp_missing = temperature.isna().all(axis=1)
    temp_valid = temperature.notna().all(axis=1)
    core_valid = frame[["W", "P", "R", "ON"]].notna().all(axis=1)
    producing = r.gt(300) & p.gt(50) & on.gt(0)
    starts = producing & ~producing.shift(fill_value=False)
    stops = ~producing & producing.shift(fill_value=False)
    start_count = starts.rolling(35).sum()
    stop_count = stops.rolling(35).sum()
    continuous = core_valid.rolling(36).sum().eq(36)
    recently_idle = r.shift(6).lt(100) & p.shift(6).lt(30)
    steady_heat = run & lo.P.gt(300) & band.P.lt(180) & band.R.lt(120) & band.A.lt(1.5)
    stable_grid_load = run & lo.P.gt(300) & band.P.lt(200) & band.Q.lt(40)
    enough = frame.notna().rolling(36).sum().min(axis=1).eq(36)
    hourly = (frame.index.minute == 0) & (np.arange(len(frame)) >= 35)
    hot = frame[GROUPS["temp"]].max(axis=1, skipna=False)
    winding_change = change[GROUPS["temp"]]
    error_active = err.gt(0)
    error_onsets = error_active & err.shift().eq(0)
    separated = t.gt(1.5)
    separation_onsets = separated & t.shift().le(1.5)
    specs = {
        "rated_pitch_response": (
            run
            & lo.P.gt(2230)
            & hi.P.lt(2370)
            & lo.C.ge(2200)
            & change.W.gt(1.5)
            & change.B.gt(1.5),
            change.W,
        ),
        "below_rated_reference": (
            run
            & lo.C.gt(250)
            & hi.C.lt(1800)
            & band.C.lt(10)
            & change.W.gt(1.5)
            & shortfall.abs().rolling(7).max().lt(120),
            change.W,
        ),
        "wind_associated_ramp": (
            run
            & lo.P.gt(100)
            & hi.P.lt(2120)
            & lo.C.ge(2250)
            & band.C.lt(10)
            & change.W.gt(1.3)
            & change.P.gt(500),
            change.P,
        ),
        "falling_power_with_wind": (
            run
            & lo.P.gt(100)
            & lo.C.ge(2200)
            & band.C.lt(10)
            & change.W.lt(-1.2)
            & change.P.lt(-500),
            -change.P,
        ),
        "reference_reduction_and_power_drop": (
            run
            & change.C.lt(-250)
            & change.P.lt(-250)
            & band.W.lt(2)
            & shortfall.abs().lt(120)
            & shortfall.shift(6).abs().lt(120),
            -change.C,
        ),
        "multiple_start_stop_transitions": (
            continuous & start_count.ge(2) & stop_count.ge(2) & r.lt(100) & p.lt(30),
            start_count + stop_count,
        ),
        "generation_after_repeated_starts": (
            continuous & start_count.ge(2) & stop_count.ge(1) & run & lo.P.gt(100),
            start_count + stop_count,
        ),
        "ready_without_generation": (
            continuous & lo.READY.eq(600) & hi.R.lt(100) & hi.P.lt(30) & hi.ON.eq(0),
            lo.W,
        ),
        "sustained_generation_after_idle": (
            continuous
            & recently_idle
            & r.gt(300)
            & p.gt(100)
            & on.rolling(3).min().eq(600)
            & p.rolling(3).min().gt(100),
            p,
        ),
        "stop_with_error_flag": (
            continuous
            & r.shift(6).gt(300)
            & p.shift(6).gt(150)
            & r.lt(100)
            & p.lt(30)
            & err.gt(0)
            & on.eq(0),
            p.shift(6),
        ),
        "winding_spread_widens": (
            steady_heat
            & t.diff(6).gt(0.5)
            & hot.diff(6).gt(0.75)
            & winding_change.abs().max(axis=1).lt(6),
            t.diff(6),
        ),
        "windings_warm_with_load": (
            run
            & change.P.gt(300)
            & change[GROUPS["temp"]].min(axis=1).gt(2)
            & t.diff(6).abs().lt(2),
            change[GROUPS["temp"]].mean(axis=1),
        ),
        "windings_cool_as_load_falls": (
            change.P.lt(-380) & change[GROUPS["temp"]].max(axis=1).lt(-1.5),
            -change[GROUPS["temp"]].mean(axis=1),
        ),
        "stable_winding_temperature_offset": (
            steady_heat & t.rolling(7).min().gt(1) & t.diff(6).abs().lt(0.75),
            t,
        ),
        "one_winding_moves_more": (
            steady_heat
            & winding_change.abs().max(axis=1).gt(0.8)
            & winding_change.abs().median(axis=1).lt(0.75),
            t.diff(6).abs(),
        ),
        "steady_phase_current_spread": (
            stable_grid_load & i.rolling(7).min().gt(2) & v.rolling(7).max().lt(9),
            i.rolling(7).min(),
        ),
        "phase_spreads_move_differently": (
            run & i.diff(6).gt(2) & v.diff(6).abs().lt(1.5),
            i.diff(6),
        ),
        "current_spread_eases_during_power_change": (
            run & i.rolling(6).max().gt(i + 2) & change.P.abs().gt(250),
            i.rolling(6).max() - i,
        ),
        "grid_phase_voltage_spread_increases": (
            run & v.diff(6).gt(1) & i.diff(6).abs().lt(3),
            v.diff(6),
        ),
        "low_phase_spread_at_load": (
            run & lo.P.gt(1500) & i.rolling(7).max().lt(4) & v.rolling(7).max().lt(5),
            lo.P,
        ),
        "all_telemetry_missing_at_cutoff": (
            missing & missing.rolling(36).sum().between(1, 30),
            missing.rolling(36).sum(),
        ),
        "reporting_resumes_after_gap": (
            core_valid
            & missing.shift(1, fill_value=False)
            & core_valid.rolling(36).sum().ge(6),
            missing.rolling(36).sum(),
        ),
        "earlier_temperature_gaps": (
            core_valid & temp_valid & temp_missing.rolling(36).sum().between(1, 24),
            temp_missing.rolling(36).sum(),
        ),
        "near_flat_wind_while_generating": (
            near_flat_w & run & lo.P.gt(100) & band.P.gt(100) & lo.W.gt(0),
            band.P,
        ),
        "fresh_zero_running_time": (
            core_valid & hi.ON.eq(0) & hi.R.lt(100) & hi.P.lt(30) & lo.W.gt(0),
            band.W,
        ),
        "recurring_error_flag_after_generation": (
            continuous
            & error_active
            & error_onsets.rolling(35).sum().ge(2)
            & stop_count.ge(1)
            & p.lt(30),
            error_onsets.rolling(35).sum(),
        ),
        "repeated_thermal_separation": (
            steady_heat
            & separated
            & separated.rolling(36).sum().ge(8)
            & separation_onsets.rolling(35).sum().ge(2),
            t,
        ),
        "current_spread_under_steady_load": (
            stable_grid_load & i.rolling(7).min().gt(1.5),
            i.rolling(7).min(),
        ),
        "live_operation_check_telemetry": (
            core_valid
            & temp_valid
            & temp_missing.rolling(36).sum().ge(1)
            & on.eq(600)
            & p.gt(100),
            temp_missing.rolling(36).sum(),
        ),
        "thermal_handover_after_output_loss": (
            err.gt(0)
            & p.lt(30)
            & r.lt(100)
            & p.shift(6).gt(300)
            & hot.notna()
            & hot.diff(6).abs().gt(1),
            hot,
        ),
    }
    allow_gaps = {
        "all_telemetry_missing_at_cutoff",
        "reporting_resumes_after_gap",
        "earlier_temperature_gaps",
        "live_operation_check_telemetry",
    }
    for label, (valid, score) in specs.items():
        mask = hourly & valid
        if label not in allow_gaps:
            mask &= enough
        yield label, score[mask]


def build(directory, metadata):
    turbines = pd.read_csv(metadata).set_index("Station ID")
    candidates = {split: {label: [] for label in ANSWERS} for split in SPLITS}
    for split, months in SPLITS.items():
        for month in months:
            print(f"Screening {month}", flush=True)
            for station, frame in read_month(directory, month).items():
                if station not in turbines.index:
                    continue
                for label, scores in select(frame):
                    # Retain the top windows per turbine and month to allow de-overlapping.
                    chosen = scores.nlargest(6)
                    for cutoff, score in chosen.items():
                        candidates[split][label].append(
                            (score, station, cutoff, month, frame.loc[:cutoff].tail(36))
                        )

    return candidates, turbines


def choose(candidates, turbines):
    used = {}
    for label, (family, preferred_split, render) in ANSWERS.items():
        alternatives = (
            preferred_split,
            *(split for split in SPLITS if split != preferred_split),
        )
        for split in alternatives:
            ranked = sorted(
                candidates[split][label], key=lambda row: (-row[0], row[1], row[2])
            )
            for _, station, cutoff, month, frame in ranked:
                if any(
                    abs(cutoff - other) < pd.Timedelta(hours=6)
                    for other in used.get(station, [])
                ):
                    continue
                pair = render(frame)
                if pair is None:
                    continue
                used.setdefault(station, []).append(cutoff)
                turbine = turbines.loc[station]
                yield (
                    family,
                    label,
                    split,
                    month,
                    int(station),
                    turbine["Turbine Name"],
                    int(turbine["Rated power (kW)"]),
                    frame,
                    pair,
                )
                break
            else:
                continue
            break
        else:
            raise ValueError(f"No nonoverlapping source window found for {label}")


def write_examples(output, selected):
    rows = []
    for family, label, split, month, station, turbine, rating, frame, pair in selected:
        cutoff = frame.index[-1].strftime("%Y-%m-%dT%H:%M:%SZ")
        window_id = f"{turbine}_{cutoff}"
        sensors = frame[list(CHANNELS)].T
        context = {
            "turbine_id": turbine,
            "cutoff_utc": cutoff,
            "rated_capacity_kw": rating,
            "alarm_status": None,
            "controller_status": None,
            "operating_rules": None,
            "matched_baseline": None,
            "independent_evidence": None,
        }
        series = sensors.to_numpy()
        values = [
            [None if np.isnan(value) else float(value) for value in channel]
            for channel in series
        ]
        for stage, (question, answer) in zip(("reasoning", "qa"), pair, strict=True):
            rows.append(
                {
                    "id": f"{window_id}_{stage}",
                    "window_id": window_id,
                    "stage": stage,
                    "family": family,
                    "scenario": label,
                    "split": split,
                    "source": {"record": RECORD, "month": month, "station_id": station},
                    "input": {
                        "question": question,
                        "context": context,
                        "values": values,
                        "valid": sensors.notna().to_numpy().tolist(),
                    },
                    "answer": answer,
                }
            )
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(
                json.dumps(
                    row, ensure_ascii=False, allow_nan=False, separators=(",", ":")
                )
                + "\n"
            )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path)
    parser.add_argument("metadata", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_name("examples.jsonl")
    )
    args = parser.parse_args()
    candidates, turbines = build(args.archives, args.metadata)
    for label in ANSWERS:
        print(
            label,
            *(f"{split}={len(items[label])}" for split, items in candidates.items()),
        )
    selected = list(choose(candidates, turbines))
    rows = write_examples(args.output, selected)
    print(f"Wrote {len(rows)} annotations to {args.output}")


if __name__ == "__main__":
    main()
