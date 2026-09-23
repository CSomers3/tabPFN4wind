import numpy as np
import pandas as pd
import pytest
from windpfn import data, dataset, nwp

T = lambda s: pd.Timestamp(s, tz="UTC")


@pytest.mark.parametrize("day,n", [("2025-03-30", 46), ("2025-10-26", 50), ("2025-06-01", 48)])
def test_fuelhh_clock_days(day, n):
    o = data.fuelhh("2025-03-01", "2025-11-01")
    assert (o.index.tz_convert("Europe/London").date == pd.Timestamp(day).date()).sum() == n


def test_hourly_is_hour_ending():
    o = pd.Series([1.0, 3.0, 5.0], pd.date_range(T("2025-01-01 00:00"), periods=3, freq="30min"))
    assert dataset.hourly(o)[T("2025-01-01 01:00")] == 2.0


def test_issue_times_follow_clock():
    i = data.issues("2025-03-25", "2025-04-02")
    assert (i.dt.strftime("%H:%M")[:"2025-03-30"] == "07:30").all()
    assert (i.dt.strftime("%H:%M")["2025-03-31":] == "08:30").all()
    assert (i < i.index).all()


def test_run_for_uses_publication_time():
    # 00Z of 2025-01-13 was public at 08:34Z, so a 07:30Z issue must fall back to the 18Z run
    run, pub, _ = nwp.run_for(T("2025-01-13 07:30"), T("2025-01-14"))
    assert run == T("2025-01-12 18:00") and pub <= T("2025-01-13 07:30")


def test_check_catches_leak():
    df = dataset.build("2025-03-29", "2025-03-30")
    dataset.check(df)
    with pytest.raises(AssertionError):
        dataset.check(df.assign(published=df.issue_time + pd.Timedelta('1min')))
    with pytest.raises(AssertionError):
        dataset.check(df.assign(windfor_pub=df.issue_time))


def test_target_is_what_windfor_forecasts():
    df = dataset.build("2024-10-01", "2024-11-30")
    slope = lambda y: np.polyfit(df.windfor, df[y], 1)[0]
    assert 0.95 < slope("y") < 1.05 and slope("y_metered") < 0.9
