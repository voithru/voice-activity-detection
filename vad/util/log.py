import logging
from datetime import datetime

from pytz import timezone


def setup_logger():
    logger = logging.root
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    log_format = "[%(levelname)s] %(asctime)s %(name)s > %(message)s"
    datetime_format = "%Y-%m-%dT%H:%M:%S %Z"
    handler.setFormatter(Formatter(log_format, datefmt=datetime_format))
    logger.addHandler(handler)


class Formatter(logging.Formatter):
    """override logging.Formatter to use an aware datetime object"""

    def converter(self, timestamp):
        return datetime.fromtimestamp(timestamp, tz=timezone("Asia/Seoul"))

    def formatTime(self, record, datefmt=None):
        t = self.converter(record.created)
        if datefmt:
            s = t.strftime(datefmt)
        else:
            try:
                s = t.isoformat(timespec="milliseconds")
            except TypeError:
                s = t.isoformat()
        return s
