from pathlib import Path
from pyGTFSHandler.models import StopTimes


class Feed:
    def __init__(self, gtfs_dir: list[str | Path]):
        if not isinstance(gtfs_dir, list):
            raise TypeError("paths must be a list of str or Path objects")
        if not all(isinstance(p, (str, Path)) for p in gtfs_dir):
            raise TypeError("all elements in paths must be str or Path")

        self.gtfs_dir = [Path(p) for p in gtfs_dir]
        for p in self.gtfs_dir:
            if not p.is_dir():
                raise ValueError(f"{p} is not a valid directory.")

        self.stop_times = StopTimes(self)

    def gtfs_dirs(self):
        return self.gtfs_dir
