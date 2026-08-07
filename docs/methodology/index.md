# Technical Explanations / Methodology

The non-obvious parts of `pyGTFSHandler`, grounded directly in the source code
(and, for direction/headway, in the two dedicated methodology notebooks):

- [Calendar handling](calendar.md) — how `calendar.txt`/`calendar_dates.txt`
  are resolved into "active service_ids on date X", and how multi-day/
  overnight trips are handled without duplicating rows.
- [Frequency-based trip expansion](frequencies.md) — the headway math that
  turns `frequencies.txt` rows into concrete stop_times departures.
- [Direction & headway methodology](direction-headway.md) — how geometric
  `direction_id` is derived from shape bearings, and the RMS-style headway
  formula used to summarize departure regularity.
- [Edge & filtering logic](edges-filtering.md) — the distance/geometry math
  behind edge (stop-to-stop segment) construction and the feed-wide
  date/time/route/AOI filter pipeline.
