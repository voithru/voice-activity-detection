from datetime import datetime, timedelta

from pysrt import SubRipTime


def parse_timecode_to_timedelta(timecode: str) -> timedelta:
    epoch = datetime(year=1900, month=1, day=1)
    return datetime.strptime(timecode, "%H:%M:%S.%f") - epoch


def parse_time_dict_to_timedelta(t: dict) -> timedelta:
    return timedelta(
        hours=t["hours"], minutes=t["minutes"], seconds=t["seconds"], milliseconds=t["milliseconds"]
    )


def format_timedelta_to_timecode(t: timedelta):
    time_dict = format_timedelta_to_time_dict(t)
    hours = time_dict["hours"]
    minutes = time_dict["minutes"]
    seconds = time_dict["seconds"]
    milliseconds = time_dict["milliseconds"]
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def format_timedelta_to_time_dict(t: timedelta) -> dict:
    total_seconds = int(t.total_seconds())
    hours = total_seconds // 3600
    minutes = total_seconds % 3600 // 60
    seconds = total_seconds % 60
    milliseconds = round(t.microseconds / 1000)
    return {"hours": hours, "minutes": minutes, "seconds": seconds, "milliseconds": milliseconds}


def format_timedelta_to_milliseconds(t: timedelta) -> int:
    return int(t.total_seconds() * 1000)


def subrip_time_to_timedelta(subrip_time: SubRipTime) -> timedelta:
    return timedelta(
        hours=subrip_time.hours,
        minutes=subrip_time.minutes,
        seconds=subrip_time.seconds,
        milliseconds=subrip_time.milliseconds,
    )


def format_timedelta_to_subrip_time(t: timedelta) -> SubRipTime:
    total_seconds = int(t.total_seconds())
    hours = total_seconds // 3600
    minutes = total_seconds % 3600 // 60
    seconds = total_seconds % 60
    milliseconds = round(t.microseconds / 1000)
    return SubRipTime(hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds)
