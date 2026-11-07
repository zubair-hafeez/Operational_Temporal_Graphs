"""Causal, development-only cohort adapters for bounded cross-domain pilots.

Event rows before each origin determine attributes and memberships. Terminal
events attach labels separately. Future rows are discarded before attributes
are encoded. Final periods are not prepared by this script.
"""
from __future__ import annotations
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/extension_20261021"
OUT = ROOT / "data/processed/extension_20261021"
REPORT = ROOT / "reports/extension_20261021"
DAY = 86400.


def stamp(text):
    return pd.Timestamp(text, tz="UTC").timestamp()


def seconds(values, dayfirst=False):
    times = pd.to_datetime(values, format="mixed", dayfirst=dayfirst, utc=True, errors="coerce")
    return times.astype("int64").to_numpy() / 1e9


def read_zip(name, member, **kwargs):
    with zipfile.ZipFile(RAW / name) as z:
        return pd.read_csv(z.open(member), **kwargs)


def interval_events(frame, cats, nums, relations, end):
    """Keep one entry event per entity; end time is a label only."""
    f = frame.loc[(frame.start < end) & (frame.start < frame.end)].copy()
    f["end"] = np.where(f.end < end, f.end, np.inf)
    f["time"] = f.start
    f["state"] = "active"
    return f, cats + ["state"], nums, relations


def load(source, evaluation=False):
    audit = {"source": source, "stage": "exploratory development pilot", "future_evaluation_scored": False}
    if source.startswith("azure"):
        full = source == "azure_full"
        names = ["id", "subscription", "deployment", "start", "end", "max_cpu", "avg_cpu", "p95_cpu", "category", "cores", "memory"]
        if full:
            # Select whole subscriptions by a predeclared hash, before looking at outcomes.
            parts = []
            for chunk in pd.read_csv(RAW / "azure_v2_vmtable.csv.gz", names=names, chunksize=100_000):
                keep = chunk.subscription.map(lambda s: int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], "big") % 20 == 0)
                parts.append(chunk.loc[keep & (chunk.start < (30 if evaluation else 20) * DAY)].copy())
            f = pd.concat(parts, ignore_index=True)
        else:
            f = pd.read_csv(RAW / "azure_v2_vmtable_prefix.csv", names=names)
        audit.update(raw_rows=len(f), sample="whole subscriptions with SHA256 first32bits mod20=0" if full else "biased row prefix; sampled-population pilot only",
                     forbidden_features=["max_cpu", "avg_cpu", "p95_cpu", "category", "end"],
                     category_warning="workload class may be derived from lifetime use; excluded",
                     zero_duration_rows=int((f.start == f.end).sum()), negative_intervals=int((f.end < f.start).sum()))
        events, cats, nums, rel = interval_events(f, ["cores", "memory"], [], ["deployment", "subscription"], (30 if evaluation else 20) * DAY)
        bounds = [0., 10*DAY, 13*DAY, 16*DAY, 20*DAY]
        horizons = np.array([5, 30, 120, 360]) * 60.
        origins = np.arange(DAY, 20*DAY, 6*3600.)
        audit["target"] = "observed VM deletion within horizon, active sampled VMs"
    elif source == "citibike":
        f = read_zip("JC-202401-citibike-tripdata.csv.zip", "JC-202401-citibike-tripdata.csv")
        f = f.rename(columns={"ride_id": "id", "start_station_id": "station"})
        # January has no DST transition; official local timestamps use New York time.
        for key, col in [("start", "started_at"), ("end", "ended_at")]:
            f[key] = pd.to_datetime(f[col]).dt.tz_localize("America/New_York").astype("int64") / 1e9
        start = stamp("2024-01-01 05:00")
        events, cats, nums, rel = interval_events(f, ["rideable_type", "member_casual"], [], ["station"], start + 20*DAY)
        bounds = [start + k*DAY for k in [0, 10, 13, 16, 20]]
        horizons = np.array([5, 15, 30, 60]) * 60.
        origins = np.arange(start + DAY, bounds[-1], 3600.)
        # Keep all hours, including low-load cohorts; no outcome-based selection.
        audit.update(raw_rows=len(f), target="recorded trip end for rides active at origin", forbidden_features=["end_station_id", "end_lat", "end_lng", "ended_at"],
                     sampling_limitation="publisher-filtered trips, including retrospective removal of short/staff trips; population is released rides")
    elif source in ["airline", "airline_multisite"]:
        name = "On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_2024_1.csv"
        columns = ["FlightDate", "Reporting_Airline", "Origin", "Dest", "CRSDepTime", "DepDelay", "Cancelled", "Distance", "CRSElapsedTime"]
        f = read_zip("bts_202401.zip", name, usecols=columns)
        raw_rows = len(f)
        zones={"JFK":"America/New_York", "LGA":"America/New_York", "EWR":"America/New_York"}
        if source == "airline_multisite":
            zones.update({"ATL":"America/New_York", "MIA":"America/New_York", "ORD":"America/Chicago", "DFW":"America/Chicago", "DEN":"America/Denver", "LAX":"America/Los_Angeles", "SFO":"America/Los_Angeles", "SEA":"America/Los_Angeles"})
        f = f.loc[f.Origin.isin(zones) & (f.FlightDate < "2024-01-21")].copy()
        f["timezone"]=f.Origin.map(zones)
        f["start"]=0.
        for zone,g in f.groupby("timezone"):
            f.loc[g.index,"start"] = pd.to_datetime(g.FlightDate).dt.tz_localize(zone).astype("int64") / 1e9
        scheduled = (f.CRSDepTime // 100)*60 + f.CRSDepTime % 100
        f["end"] = f.start + (scheduled + f.DepDelay)*60
        f.loc[f.Cancelled == 1, "end"] = np.inf
        if f.end.isna().any():
            raise ValueError("Uncancelled flights without departure labels require a censoring policy")
        f["scheduled_minutes"] = scheduled
        f["id"] = np.arange(len(f))
        f = f.rename(columns={"Origin":"airport", "Reporting_Airline":"carrier", "Dest":"destination"})
        start = stamp("2024-01-01 05:00")
        events, cats, nums, rel = interval_events(f, ["timezone"] if source == "airline_multisite" else [], ["scheduled_minutes", "Distance", "CRSElapsedTime"], ["airport", "carrier", "destination"], start+20*DAY)
        bounds = [start+k*DAY for k in [0,10,13,16,20]]
        horizons = np.array([.5,1,3,6])*3600.
        origins = np.arange(start+DAY, bounds[-1], 6*3600.)
        audit.update(raw_rows=raw_rows, selected_flights=len(f), target="departure by common future horizons among today's scheduled flights still undeparted at origin",
                     assumption="reported schedule treated as available at local midnight; archive does not prove schedule version availability",
                     cancellation="included as no departure within horizon; cancellation status is label-only",
                     forbidden_features=["DepDelay", "Cancelled", "arrival fields", "actual aircraft assignment", "cause of delay"],
                     scope="scheduled workload population, not in-flight arrivals; development only", airports=zones,
                     sampling="NY airports" if source == "airline" else "prespecified 11-airport geographically dispersed pilot; selected before looking at its scores")
    elif source in ["uci", "bpi2014"]:
        if source == "uci":
            f = read_zip("uci_incidents.zip", "incident_event_log.csv", na_values="?", low_memory=False)
            f = f.rename(columns={"number": "id", "incident_state": "state", "assigned_to": "handler", "assignment_group": "team"})
            f["time"] = seconds(f.sys_updated_at, True)
            # Updates, rather than backfilled open/resolution times, define visibility.
            cats = ["state", "priority", "contact_type"]
            nums = ["reassignment_count", "reopen_count", "sys_mod_count"]
            rel = ["handler", "team"]
            terminal = ["Resolved", "Closed"]
            bounds = [stamp(s) for s in ["2016-03-01", "2016-03-22", "2016-04-01", "2016-04-15", "2016-05-01"]]
            audit.update(target="first observed Resolved/Closed update after first observed nonterminal update",
                         forbidden_features=["resolved_at", "closed_at", "resolved_by", "closed_code", "made_sla", "rfc", "caused_by"],
                         semantics="event-time replay of recorded updates, not true resolution timestamp; completed-case extraction may bias population")
        else:
            f = read_zip("BPIC-2014.zip", "Detail Incident Activity.csv", sep=";")
            f = f.rename(columns={"Incident ID": "id", "IncidentActivity_Type": "state", "Assignment Group": "team"})
            f["time"] = seconds(f.DateStamp, True)
            cats, nums, rel, terminal = ["state"], [], ["team"], ["Closed"]
            bounds = [stamp(s) for s in ["2013-10-01", "2013-11-01", "2013-11-15", "2013-12-01", "2014-01-01"]]
            audit.update(target="first Closed activity after observed Open activity", forbidden_features=["final incident table attributes", "cause CI", "final related change", "KM number"],
                         semantics="retain only entities with observed Open; first closure, later reopening outside target")
        audit["raw_rows"] = len(f)
        observed_end = stamp("2016-06-01" if source == "uci" else "2014-02-01") if evaluation else bounds[-1]
        f = f.loc[(f.time < observed_end) & (f.time > 0)].copy().sort_values("time", kind="stable")
        starts = f.loc[f.state.eq("Open")].groupby("id").time.min() if source == "bpi2014" else f.loc[~f.state.isin(terminal)].groupby("id").time.min()
        ends = f.loc[f.state.isin(terminal)].groupby("id").time.min()
        f["start"] = f.id.map(starts)
        f["end"] = f.id.map(ends).fillna(np.inf)
        f = f.loc[f.start.notna() & (f.start < f.end) & (f.time >= f.start)].copy()
        events = f
        horizons = np.array([1, 6, 24, 72]) * 3600.
        origins = np.arange(bounds[0] + DAY/2, bounds[-1], DAY)
    else:
        raise ValueError(source)
    if evaluation:
        if source == "azure_full":
            origins = np.arange(22*DAY, 29*DAY, 6*3600.)
        elif source in ["uci", "bpi2014"]:
            first = stamp("2016-05-01" if source == "uci" else "2014-01-01")+DAY/2
            origins = first+np.arange(28)*DAY
        else:
            raise ValueError("Only prespecified confirmation sources may be opened")
        audit["stage"] = "frozen-protocol confirmation"
        audit["future_evaluation_scored"] = False
    events = events.sort_values(["time", "id"], kind="stable").reset_index(drop=True)
    events["event_count"] = events.groupby("id").cumcount() + 1
    events["entity"] = pd.factorize(events.id, sort=True)[0]
    for c in cats:
        events[c] = events[c].fillna("missing").astype(str)
    for c in nums:
        events[c] = pd.to_numeric(events[c], errors="coerce").fillna(0)
    for c in rel:
        events[c] = pd.factorize(events[c], sort=True)[0]
    return events, cats, nums, rel, np.array(bounds), horizons, origins, audit


def build(source, evaluation=False):
    out = OUT / (source+"_final" if evaluation else source)
    out.mkdir(parents=True, exist_ok=False)
    f, cats, nums, rel, bounds, horizons, origins, audit = load(source, evaluation)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=5)
    enc.fit(f.loc[f.time < bounds[1], cats])
    cat_x = enc.transform(f[cats]).astype(np.float32)
    fixed_x = np.column_stack([cat_x, np.log1p(np.clip(f[nums].to_numpy(float), 0, None))]).astype(np.float32)
    f["row"] = np.arange(len(f))
    all_x, all_y, all_r, all_t, all_e, all_h = [], [], [], [], [], []
    metadata = []
    for origin in origins:
        part = "final" if evaluation else next((name for i, name in enumerate(["train", "calibration", "dependence", "validation"])
                     if bounds[i] <= origin and origin + horizons[-1] < bounds[i+1]), None)
        if part is None:
            continue
        prefix = f.loc[f.time <= origin]
        current = prefix.drop_duplicates("entity", keep="last")
        current = current.loc[(current.start <= origin) & (current.end > origin)].copy()
        if source in ["airline", "airline_multisite"]:
            current = current.loc[origin < current.start+DAY]
        if len(current) < 8:
            continue
        assert current.time.max() <= origin
        phase = 2*np.pi*(origin % DAY)/DAY
        week = 2*np.pi*((origin // DAY + 3) % 7)/7
        x = np.column_stack([np.log1p((origin-current.start)/3600), np.log1p((origin-current.time)/3600),
                             np.log1p(current.event_count), fixed_x[current.row],
                             np.full(len(current), np.sin(phase)), np.full(len(current), np.cos(phase)),
                             np.full(len(current), np.sin(week)), np.full(len(current), np.cos(week))]).astype(np.float32)
        # Relation summaries and event-history tokens use the observed prefix only.
        history = np.zeros((len(current), len(rel), 8, 5), np.float32)
        for j, col in enumerate(rel):
            latest = {}
            recent = prefix.loc[prefix.time >= origin - 7*DAY]
            for key, g in recent.loc[recent[col] >= 0].groupby(col, sort=False):
                g = g.tail(8)
                lag = (origin - g.time.to_numpy())/3600.
                age = np.maximum(0, g.time.to_numpy()-g.start.to_numpy())/3600.
                # Terminal indicator is an observed event, never an attached future label.
                terminal = g.state.isin(["Closed", "Resolved"]).to_numpy(float)
                token = np.column_stack([np.log1p(lag), np.log1p(age), np.log1p(g.event_count), terminal, np.ones(len(g))])
                padded = np.zeros((8, 5), np.float32); padded[-len(g):] = token
                latest[key] = padded
            for row, key in enumerate(current[col]):
                if key in latest:
                    history[row, j] = latest[key]
        y = ((current.end.to_numpy()[:, None]-origin) <= horizons[None]).astype(np.float32)
        offset = sum(len(v) for v in all_x)
        metadata.append({"origin": float(origin), "part": part, "offset": offset, "size": len(current)})
        all_x.append(x); all_y.append(y); all_r.append(current[rel].to_numpy(np.int64)); all_t.append(np.full(len(x), origin))
        all_e.append(current.entity.to_numpy(np.int64)); all_h.append(history)
    if not all_x:
        raise RuntimeError("No eligible cohorts")
    np.savez_compressed(out / "cohorts.npz", x=np.concatenate(all_x), y=np.concatenate(all_y),
                        relations=np.concatenate(all_r), time=np.concatenate(all_t), entity=np.concatenate(all_e), history=np.concatenate(all_h))
    audit.update(bounds=bounds.tolist(), horizons_seconds=horizons.tolist(), relationships=rel, cohorts=metadata,
                 queries=sum(len(x) for x in all_x), features=int(all_x[0].shape[1]),
                 categorical_columns=cats, numeric_columns=nums, vocabulary_fit_before=float(bounds[1]),
                 history="last 8 observed relation events in prior 7 days; lag, event age, count, observed terminal, mask",
                 split_rule="whole longest horizon strictly before partition end; train/calibration/dependence/validation separate",
                 code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out / "manifest.json").write_text(json.dumps(audit, indent=2)+"\n")
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / f"prepared_{source}{'_final' if evaluation else ''}.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({k:v for k,v in audit.items() if k != "cohorts"}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["azure", "azure_full", "citibike", "airline", "airline_multisite", "uci", "bpi2014"], required=True)
    build(ap.parse_args().source)
