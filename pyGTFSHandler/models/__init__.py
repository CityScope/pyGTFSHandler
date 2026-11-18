"""One class per GTFS table (stops, stop times, trips, routes, calendar, shapes, frequencies), each holding a Polars LazyFrame."""
from .stop_times import StopTimes
from .stops import Stops
from .trips import Trips
from .calendar import Calendar
from .routes import Routes
from .shapes import Shapes

__all__ = [
    "StopTimes",
    "Stops",
    "Trips",
    "Calendar",
    "Routes",
    "Shapes",
]
