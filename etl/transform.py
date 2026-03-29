import xarray as xr
import numpy as np
import pandas as pd

def decode_bytes(val) -> str:
    if isinstance(val, bytes):
        return val.decode("utf-8").strip()
    return str(val).strip()

def clean_qc(val) -> str | None:
    """Return single valid QC char or None — never 'nan' or multi-char strings."""
    s = decode_bytes(val)
    if s in ("", "nan", "NaN", "None"):
        return None
    return s[0] if len(s) >= 1 else None   # take only first character

def parse_juld(juld_val):
    try:
        ts = pd.Timestamp(juld_val)
        if pd.isnull(ts):
            return None
        return ts.to_pydatetime()
    except Exception:
        return None

def transform(nc_file_path: str) -> dict:
    print(f"[TRANSFORM] Reading {nc_file_path}...")
    ds = xr.open_dataset(nc_file_path, decode_times=True)

    try:
        wmo_id        = decode_bytes(ds["PLATFORM_NUMBER"].values[0])
        dac           = decode_bytes(ds["DATA_CENTRE"].values[0])
        platform_type = decode_bytes(ds["PLATFORM_TYPE"].values[0])
        institution   = decode_bytes(ds["PI_NAME"].values[0]) if "PI_NAME" in ds else None

        float_info = {
            "wmo_id":        wmo_id,
            "dac":           dac or None,
            "institution":   institution or None,
            "platform_type": platform_type or None,
        }

        print(f"[TRANSFORM] Float: {wmo_id} | DAC: {dac} | Platform: {platform_type}")

        n_prof   = ds.sizes["N_PROF"]   # fix FutureWarning — use .sizes not .dims
        profiles = []

        for i in range(n_prof):
            profile_date = parse_juld(ds["JULD"].values[i])
            if profile_date is None:
                continue

            lat = float(ds["LATITUDE"].values[i])
            lon = float(ds["LONGITUDE"].values[i])

            if np.isnan(lat) or np.isnan(lon):
                continue
            if abs(lat) > 90 or abs(lon) > 180:
                continue

            cycle     = int(ds["CYCLE_NUMBER"].values[i])
            direction = clean_qc(ds["DIRECTION"].values[i])
            pres_qc   = clean_qc(ds["PROFILE_PRES_QC"].values[i])  # was getting 'nan'

            pres = ds["PRES"].values[i].astype(float)
            temp = ds["TEMP"].values[i].astype(float)
            psal = ds["PSAL"].values[i].astype(float)

            pres_qc_arr = [clean_qc(c) for c in ds["PRES_QC"].values[i]]
            temp_qc_arr = [clean_qc(c) for c in ds["TEMP_QC"].values[i]]
            psal_qc_arr = [clean_qc(c) for c in ds["PSAL_QC"].values[i]]

            measurements = []
            for lvl in range(len(pres)):
                p = pres[lvl]
                if np.isnan(p) or p <= 0:
                    continue

                measurements.append({
                    "depth_level": lvl,
                    "pressure":    round(float(p), 4),
                    "pres_qc":     pres_qc_arr[lvl] if lvl < len(pres_qc_arr) else None,
                    "temperature": round(float(temp[lvl]), 4) if not np.isnan(temp[lvl]) else None,
                    "temp_qc":     temp_qc_arr[lvl] if lvl < len(temp_qc_arr) else None,
                    "salinity":    round(float(psal[lvl]), 4) if not np.isnan(psal[lvl]) else None,
                    "sal_qc":      psal_qc_arr[lvl] if lvl < len(psal_qc_arr) else None,
                })

            profiles.append({
                "cycle_number":    cycle,
                "profile_date":    profile_date,
                "latitude":        lat,
                "longitude":       lon,
                "direction":       direction,
                "profile_pres_qc": pres_qc,
                "source_file":     nc_file_path,
                "measurements":    measurements,
            })

        total_m = sum(len(p["measurements"]) for p in profiles)
        print(f"[TRANSFORM] {len(profiles)} profiles | {total_m} measurements")
        return {"float_info": float_info, "profiles": profiles}

    finally:
        ds.close()