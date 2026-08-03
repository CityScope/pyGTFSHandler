(function waitForMap() {
  if (!window["__MAP_VAR__"]) {
    setTimeout(waitForMap, 25);
    return;
  }
  var map = window["__MAP_VAR__"];
  var DATA = __DATA_JSON__;

  var routeTypeEmoji = DATA.route_type_emoji;
  var routeTypeEmojiFallback = DATA.route_type_emoji_fallback;
  var routeTypeName = DATA.route_type_name;
  var routeTypeNameFallback = DATA.route_type_name_fallback;
  var squareTypes = DATA.square_badge_route_types;

  // Group stop_ids by parent_station once, up front -- clicking any one
  // platform/stop_id shows the combined timetable for the whole station.
  var stopsByParent = {};
  Object.keys(DATA.stops).forEach(function (sid) {
    var p = DATA.stops[sid].parent || sid;
    (stopsByParent[p] = stopsByParent[p] || []).push(sid);
    DATA.stops[sid].metric = (DATA.stop_metrics && DATA.stop_metrics[sid]) || null;
  });

  function fmtTime(seconds) {
    if (seconds === null || seconds === undefined) return "--:--";
    var s = ((seconds % 86400) + 86400) % 86400;
    var h = Math.floor(s / 3600);
    var m = Math.floor((s % 3600) / 60);
    return (h < 10 ? "0" + h : h) + ":" + (m < 10 ? "0" + m : m);
  }

  // openable=false marks a badge as "not a generic route-open trigger" --
  // used for badges that already have their own click behavior (the
  // timetable's route-filter row, the "Filter lines..." checkboxes), so
  // clicking them doesn't *also* open a trip box (see the delegated
  // gtfs-rm-badge click handlers set up below, once openTripBox exists).
  function badgeHtml(routeId, extraClass, extraAttrs, openable) {
    var r = DATA.routes[routeId];
    if (!r) return "";
    var isSquare = squareTypes.indexOf(r.route_type) !== -1;
    var shape = isSquare ? "square" : "circle";
    var label = r.route_short_name || r.route_long_name || routeId;
    return (
      '<span class="gtfs-rm-badge ' + shape + (extraClass ? " " + extraClass : "") +
      '" data-route="' + escapeHtml(routeId) + '"' + (openable === false ? ' data-openable="false"' : "") +
      ' style="background:#' + r.route_color + ";color:#" + r.route_text_color + '" title="' +
      escapeHtml(r.route_long_name || "") + '"' + (extraAttrs ? " " + extraAttrs : "") + ">" +
      escapeHtml(label) + "</span>"
    );
  }

  function escapeHtml(s) {
    return String(s === undefined || s === null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  // Route icons are always listed most- to least-served (DATA.routes[r]
  // .service_count, computed server-side as the trip count in whichever
  // single direction runs more of them, so a round trip is never
  // double-counted).
  function sortByService(routeIds) {
    return routeIds.slice().sort(function (a, b) {
      var ca = (DATA.routes[a] && DATA.routes[a].service_count) || 0;
      var cb = (DATA.routes[b] && DATA.routes[b].service_count) || 0;
      return cb - ca;
    });
  }

  // Route icons respect the mode-filter checkboxes too: a route whose mode
  // is currently unchecked is left out of every badge list.
  function visibleRoutes(routeIds) {
    return routeIds.filter(function (r) {
      var rt = DATA.routes[r] && DATA.routes[r].route_type;
      return rt === undefined || checkedTypes[rt] !== false;
    });
  }

  function excludeRoute(routeIds, excludeId) {
    if (!excludeId) return routeIds;
    return routeIds.filter(function (r) {
      return r !== excludeId;
    });
  }

  // Average headway (mean gap between consecutive departure times) of
  // whatever set of departures is currently on screen -- recomputed
  // whenever the timetable's route-badge filter changes, so it always
  // reflects what's actually shown, not the whole unfiltered station.
  function computeAvgHeadwaySecs(deps) {
    var times = deps
      .map(function (d) { return d.dep_time; })
      .filter(function (t) { return t !== null && t !== undefined; })
      .sort(function (a, b) { return a - b; });
    if (times.length < 2) return null;
    var total = 0;
    for (var i = 1; i < times.length; i++) total += times[i] - times[i - 1];
    return total / (times.length - 1);
  }

  function fmtDuration(seconds) {
    if (seconds === null || seconds === undefined) return "--";
    var m = Math.round(seconds / 60);
    if (m < 60) return m + " min";
    var h = Math.floor(m / 60);
    var mm = m % 60;
    return h + "h" + (mm ? " " + mm + "min" : "");
  }

  function fmtSpeed(v) {
    return v === null || v === undefined || isNaN(v) ? "--" : v.toFixed(1) + " km/h";
  }

  function fmtHeadwayMin(v) {
    return v === null || v === undefined || isNaN(v) ? "--" : (v < 60 ? Math.round(v) + " min" : fmtDuration(v * 60));
  }

  // ------------------------------------------------------------------
  // Speed/headway color scales (approximating matplotlib's RdYlGn /
  // RdYlGn_r) and black-silhouette emoji recoloring for the speed/headway
  // map modes.
  // ------------------------------------------------------------------
  var COLOR_STOPS = [
    [165, 0, 38], [215, 48, 39], [244, 109, 67], [253, 174, 97],
    [254, 224, 139], [255, 255, 191], [217, 239, 139], [166, 217, 106],
    [102, 189, 99], [26, 152, 80], [0, 104, 55],
  ];
  function lerp(a, b, t) { return a + (b - a) * t; }
  function rdylgn(t) {
    t = Math.max(0, Math.min(1, t));
    var n = COLOR_STOPS.length - 1;
    var idx = Math.min(n - 1, Math.floor(t * n));
    var localT = t * n - idx;
    var c0 = COLOR_STOPS[idx];
    var c1 = COLOR_STOPS[idx + 1];
    var r = Math.round(lerp(c0[0], c1[0], localT));
    var g = Math.round(lerp(c0[1], c1[1], localT));
    var b = Math.round(lerp(c0[2], c1[2], localT));
    return "rgb(" + r + "," + g + "," + b + ")";
  }
  // speed: red (slow) -> green (fast). headway: green (frequent/low) ->
  // red (infrequent/high), i.e. the reversed scale.
  var MODE_SCALE = {
    speed: { min: 10, max: 30, reversed: false },
    headway: { min: 5, max: 60, reversed: true },
  };
  function metricColor(mode, value) {
    if (value === null || value === undefined || isNaN(value)) return "#888888";
    var scale = MODE_SCALE[mode];
    var t = (value - scale.min) / (scale.max - scale.min);
    if (scale.reversed) t = 1 - t;
    return rdylgn(t);
  }

  // Renders an emoji glyph to an offscreen canvas once and caches its PNG
  // data URL, keyed by the glyph itself -- used as a CSS mask-image so the
  // glyph's own alpha channel (its silhouette) can be recolored with any
  // background-color, regardless of the emoji font's original colors
  // ("black and white, then recolored").
  var emojiMaskCache = {};
  function emojiMaskUrl(glyph) {
    if (emojiMaskCache[glyph]) return emojiMaskCache[glyph];
    var size = 64;
    var canvas = document.createElement("canvas");
    canvas.width = size;
    canvas.height = size;
    var ctx = canvas.getContext("2d");
    ctx.font = (size * 0.82) + "px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(glyph, size / 2, size / 2 + size * 0.04);
    var url = canvas.toDataURL("image/png");
    emojiMaskCache[glyph] = url;
    return url;
  }
  // A colored, mask-recolored version of an emoji, as an inline-block span
  // with the given pixel size.
  function coloredEmojiHtml(glyph, color, sizePx) {
    var url = emojiMaskUrl(glyph);
    var style =
      "display:inline-block;width:" + sizePx + "px;height:" + sizePx + "px;" +
      "background-color:" + color + ";" +
      "-webkit-mask-image:url(" + url + ");mask-image:url(" + url + ");" +
      "-webkit-mask-size:contain;mask-size:contain;" +
      "-webkit-mask-repeat:no-repeat;mask-repeat:no-repeat;" +
      "-webkit-mask-position:center;mask-position:center;";
    return '<span class="gtfs-rm-emoji-mask" style="' + style + '"></span>';
  }

  // Current map mode: "none" (plain colored emojis, no edges), "speed" or
  // "headway" (edges + recolored stop emojis reflect that metric).
  var currentMode = "none";

  // ------------------------------------------------------------------
  // Filter control panel (mode checkboxes)
  // ------------------------------------------------------------------
  var presentTypes = [];
  Object.keys(DATA.stops).forEach(function (sid) {
    (DATA.stops[sid].modes || []).forEach(function (t) {
      if (presentTypes.indexOf(t) === -1) presentTypes.push(t);
    });
  });
  presentTypes.sort();

  var checkedTypes = {};
  presentTypes.forEach(function (t) {
    checkedTypes[t] = true;
  });

  // Per-route visibility, independent of the mode checkboxes -- set from
  // the "Filter lines..." box (defined further down, once openBox exists).
  var checkedRoutes = {};
  Object.keys(DATA.routes).forEach(function (rid) {
    checkedRoutes[rid] = true;
  });

  function refreshLiveHighlight() {
    // Route icons (beside stops) respect both filters live -- re-render the
    // current highlight so newly-hidden routes/modes disappear immediately
    // instead of only on the next click.
    if (lastHighlight) {
      highlightStops(lastHighlight.selectedIds, lastHighlight.aboveLineIds, lastHighlight.badgesBySid);
      updateMarkers();
    }
  }

  // Single-select "checkbox" group (radio inputs styled as the same
  // checkbox-list look as the mode filters) letting the user pick at most
  // one of speed/headway coloring, or none.
  var MODE_OPTIONS = [
    { key: "none", label: "None" },
    { key: "speed", label: "\U0001F3CE️ Speed" },
    { key: "headway", label: "⏱️ Headway" },
  ];

  var FilterControl = L.Control.extend({
    options: { position: "topright" },
    onAdd: function () {
      var div = L.DomUtil.create("div", "gtfs-rm gtfs-rm-filter-panel");
      L.DomEvent.disableClickPropagation(div);
      var html = "<h4>Color by</h4>";
      MODE_OPTIONS.forEach(function (opt) {
        html +=
          '<label><input type="checkbox" ' + (opt.key === "none" ? "checked" : "") +
          ' data-mode="' + opt.key + '"> ' + opt.label + "</label>";
      });
      html += "<h4>Modes</h4>";
      presentTypes.forEach(function (t) {
        var emoji = routeTypeEmoji[t] || routeTypeEmojiFallback;
        var name = routeTypeName[t] || routeTypeNameFallback;
        html +=
          '<label><input type="checkbox" checked data-type="' + t + '"> ' +
          emoji + " " + escapeHtml(name) + "</label>";
      });
      html += '<button type="button" class="gtfs-rm-lines-btn" id="gtfs-rm-lines-btn">Filter lines…</button>';
      html += '<div class="gtfs-rm-legend" id="gtfs-rm-legend"></div>';
      div.innerHTML = html;
      div.querySelectorAll("input[data-mode]").forEach(function (cb) {
        cb.addEventListener("change", function () {
          var mode = cb.getAttribute("data-mode");
          // Only one may be checked at a time: checking one unchecks the
          // rest (including forcing "none" off whenever another is on).
          div.querySelectorAll("input[data-mode]").forEach(function (other) {
            other.checked = other === cb ? cb.checked : false;
          });
          currentMode = cb.checked ? mode : "none";
          if (currentMode === "none" && mode !== "none") {
            // Unchecking the only active mode falls back to "none".
            div.querySelector('input[data-mode="none"]').checked = true;
          }
          setMode(currentMode);
        });
      });
      div.querySelectorAll("input[data-type]").forEach(function (cb) {
        cb.addEventListener("change", function () {
          checkedTypes[cb.getAttribute("data-type")] = cb.checked;
          applyFilter();
          refreshLiveHighlight();
        });
      });
      div.querySelector("#gtfs-rm-lines-btn").addEventListener("click", function () {
        openRouteFilterBox();
      });
      return div;
    },
  });
  map.addControl(new FilterControl());

  function renderLegend() {
    var el = document.getElementById("gtfs-rm-legend");
    if (!el) return;
    if (currentMode === "none") {
      el.innerHTML = "";
      return;
    }
    var scale = MODE_SCALE[currentMode];
    var stops = [];
    for (var i = 0; i <= 4; i++) {
      var t = i / 4;
      var val = scale.min + t * (scale.max - scale.min);
      stops.push(rdylgn(scale.reversed ? 1 - t : t));
    }
    var gradient = "linear-gradient(to right," + stops.join(",") + ")";
    el.innerHTML =
      '<div class="gtfs-rm-legend-title">' + (currentMode === "speed" ? "Speed (km/h)" : "Headway (min)") + "</div>" +
      '<div class="gtfs-rm-legend-bar" style="background:' + gradient + '"></div>' +
      '<div class="gtfs-rm-legend-range"><span>' + scale.min + "</span><span>" + scale.max + "</span></div>";
  }

  // Optional translucent "Google Hybrid" imagery layer, toggled on/off over
  // the existing base map (not a replacement for it) via the standard
  // Leaflet layer-switcher control, at 60% opacity so the base map still
  // shows through.
  var googleHybridLayer = L.tileLayer("https://{s}.google.com/vt/lyrs=y&x={x}&y={y}&z={z}", {
    maxZoom: 20,
    subdomains: ["mt0", "mt1", "mt2", "mt3"],
    opacity: 0.6,
    attribution: "Imagery &copy; Google",
  });
  L.control.layers(null, { "Google Hybrid": googleHybridLayer }, { collapsed: true }).addTo(map);

  // ------------------------------------------------------------------
  // Z-order panes.
  //
  // Stop markers normally live in the default `markerPane` (600). While a
  // route/stop highlight is active, markers *not* part of the highlight are
  // moved into `gtfsBelowLinePane` (549) so the highlighted route's
  // linestring (`gtfsLinePane`, 590) renders above them, while the
  // highlighted stops themselves stay in `markerPane` (600), above the
  // line. With nothing highlighted everything just sits in `markerPane`.
  // ------------------------------------------------------------------
  map.createPane("gtfsBelowLinePane");
  map.getPane("gtfsBelowLinePane").style.zIndex = 549;
  map.createPane("gtfsLinePane");
  map.getPane("gtfsLinePane").style.zIndex = 590;

  // Base draw-order ranking by mode, independent of any highlight: rail on
  // top, then subway, then tram, then any other mode, bus at the bottom.
  var MODE_Z_RANK = { "2": 4000, "1": 3000, "0": 2000, "3": 0 };
  var MODE_Z_OTHER = 1000;
  function modeBaseOffset(sid) {
    var stop = DATA.stops[sid];
    var modes = (stop && stop.modes) || [];
    if (!modes.length) return MODE_Z_OTHER;
    var best = -Infinity;
    modes.forEach(function (t) {
      var r = MODE_Z_RANK.hasOwnProperty(t) ? MODE_Z_RANK[t] : MODE_Z_OTHER;
      if (r > best) best = r;
    });
    return best;
  }

  function movePane(marker, paneName) {
    if (!marker._icon) return;
    var pane = map.getPane(paneName);
    if (pane && marker._icon.parentNode !== pane) pane.appendChild(marker._icon);
  }

  // ------------------------------------------------------------------
  // Stop markers -- plain emoji, no background "dot". `badges` (route_id
  // list) renders small line-badges beside the emoji, used while a
  // route/trip is highlighted so the relevant line(s) are visible right on
  // the stop marker.
  // ------------------------------------------------------------------
  var markers = {};
  var highlightLayer = L.layerGroup().addTo(map);
  var edgeLayer = L.layerGroup().addTo(map);

  function bindStopTooltip(marker, sid) {
    marker.unbindTooltip();
    if (currentMode === "none") return;
    var m = DATA.stop_metrics && DATA.stop_metrics[sid];
    var val = m ? m[currentMode] : null;
    var text = currentMode === "speed" ? "Speed: " + fmtSpeed(val) : "Headway: " + fmtHeadwayMin(val);
    marker.bindTooltip(text, { direction: "top", offset: [0, -10], sticky: true });
  }

  // Redraws the edge layer with the currently-selected mode's representative
  // shape geometry per edge (busiest shape for headway, fastest for speed),
  // falling back to a straight line between the two stops when no shape
  // geometry is available for that edge.
  function drawEdgeLayer() {
    edgeLayer.clearLayers();
    if (currentMode === "none" || !DATA.edges) return;
    var geomKey = currentMode === "speed" ? "geom_fast" : "geom_freq";
    Object.keys(DATA.edges).forEach(function (key) {
      var e = DATA.edges[key];
      var val = e[currentMode];
      var pts = e[geomKey];
      if (!pts || pts.length < 2) {
        var sa = DATA.stops[e.a];
        var sb = DATA.stops[e.b];
        if (!sa || !sb || sa.lat === null || sb.lat === null) return;
        pts = [
          [sa.lat, sa.lon],
          [sb.lat, sb.lon],
        ];
      }
      var color = metricColor(currentMode, val);
      var label =
        (currentMode === "speed" ? "Speed: " + fmtSpeed(val) : "Headway: " + fmtHeadwayMin(val)) +
        (e.n_trips ? " (" + e.n_trips + " trips)" : "");
      var line = L.polyline(pts, { color: color, weight: 4, opacity: 0.85 });
      line.bindTooltip(label, { sticky: true });
      line.on("mouseover", function () {
        line.setStyle({ weight: 7, opacity: 1 });
      });
      line.on("mouseout", function () {
        line.setStyle({ weight: 4, opacity: 0.85 });
      });
      line.addTo(edgeLayer);
    });
  }

  // Rebuilds every marker's icon + hover tooltip in place (used on mode
  // switch, since existing markers otherwise keep whatever icon they were
  // last given) while preserving whatever highlight/selection is active.
  function refreshAllIcons() {
    var selSet = {};
    var badgesBySid = {};
    if (lastHighlight) {
      (lastHighlight.selectedIds || []).forEach(function (id) {
        selSet[id] = true;
      });
      badgesBySid = lastHighlight.badgesBySid || {};
    }
    Object.keys(markers).forEach(function (sid) {
      var isSelected = !!selSet[sid];
      var badges = badgesBySid[sid] ? visibleRoutes(badgesBySid[sid]) : null;
      markers[sid].setIcon(makeIcon(DATA.stops[sid], isSelected, badges));
      bindStopTooltip(markers[sid], sid);
    });
  }

  function setMode(mode) {
    currentMode = mode;
    refreshAllIcons();
    drawEdgeLayer();
    renderLegend();
  }

  function stopIconHtml(stop, selected, badgeRouteIds) {
    var size = selected ? 30 : 20;
    var emojis;
    if (currentMode !== "none") {
      var m = (stop && stop.metric) || null;
      var color = metricColor(currentMode, m === null ? null : m[currentMode]);
      var glyphs = (stop.modes || []).length ? stop.modes : ["_fallback"];
      emojis = glyphs
        .map(function (t) {
          var glyph = routeTypeEmoji[t] || routeTypeEmojiFallback;
          return coloredEmojiHtml(glyph, color, size);
        })
        .join("");
    } else {
      emojis = (stop.modes || [])
        .map(function (t) {
          return routeTypeEmoji[t] || routeTypeEmojiFallback;
        })
        .join("");
      if (!emojis) emojis = routeTypeEmojiFallback;
    }
    var cls = "gtfs-rm-stop-icon" + (selected ? " gtfs-rm-selected" : "");
    var badgesHtml = "";
    if (badgeRouteIds && badgeRouteIds.length) {
      badgesHtml =
        '<span class="gtfs-rm-marker-badges">' +
        badgeRouteIds.map(function (r) { return badgeHtml(r, "gtfs-rm-marker-badge"); }).join("") +
        "</span>";
    }
    return {
      html:
        '<div class="' + cls + '" style="width:' + size + "px;height:" + size + "px;font-size:" + size + 'px;">' +
        emojis + badgesHtml + "</div>",
      size: size,
    };
  }

  function makeIcon(stop, selected, badgeRouteIds) {
    var h = stopIconHtml(stop, selected, badgeRouteIds);
    return L.divIcon({
      html: h.html,
      className: "",
      iconSize: [h.size, h.size],
      iconAnchor: [h.size / 2, h.size / 2],
    });
  }

  Object.keys(DATA.stops).forEach(function (sid) {
    var stop = DATA.stops[sid];
    if (stop.lat === null || stop.lon === null) return;
    var marker = L.marker([stop.lat, stop.lon], { icon: makeIcon(stop, false) });
    marker.gtfsStopId = sid;
    marker.setZIndexOffset(modeBaseOffset(sid));
    marker.on("click", function () {
      selectStop(sid);
    });
    bindStopTooltip(marker, sid);
    markers[sid] = marker;
    // Not added to the map here -- updateMarkers() (below, driven by zoom/
    // filter/highlight state) decides whether each stop renders as an
    // individual marker or gets folded into a cluster bubble.
  });

  // route_id -> every stop_id it serves (today), the reverse of
  // DATA.stop_routes -- used so that highlighting a route always elevates
  // *all* of its stops above the linestring and can badge all of them too,
  // not just whichever stop/trip triggered the highlight.
  var routeStops = {};
  Object.keys(DATA.stop_routes).forEach(function (sid) {
    (DATA.stop_routes[sid] || []).forEach(function (rid) {
      (routeStops[rid] = routeStops[rid] || []).push(sid);
    });
  });

  // A representative trip for a route -- whichever direction runs more
  // service (ties broken by DATA.route_trips insertion order), its
  // earliest-departing trip (route_trips is already sorted that way). Used
  // so that clicking a route icon anywhere always has *some* trip to open.
  function firstTripForRoute(routeId) {
    var prefix = routeId + "||";
    var keys = Object.keys(DATA.route_trips).filter(function (k) {
      return k.indexOf(prefix) === 0;
    });
    if (!keys.length) return null;
    keys.sort(function (a, b) {
      return (DATA.route_trips[b] || []).length - (DATA.route_trips[a] || []).length;
    });
    var trips = DATA.route_trips[keys[0]] || [];
    return trips.length ? trips[0] : null;
  }

  function openRouteTrip(routeId) {
    var tid = firstTripForRoute(routeId);
    if (tid) openTripBox(tid);
  }

  function stopVisible(sid) {
    var stop = DATA.stops[sid];
    var modeOk = (stop.modes || []).some(function (t) {
      return checkedTypes[t];
    });
    var routes = DATA.stop_routes[sid] || [];
    var routeOk = routes.length === 0 || routes.some(function (r) {
      return checkedRoutes[r] !== false;
    });
    return modeOk && routeOk;
  }

  // ------------------------------------------------------------------
  // Performance: at low zoom, thousands of individual DOM markers are
  // slow to pan/zoom. Below CLUSTER_ZOOM, filter-visible stops are grouped
  // into grid cells (in screen-pixel space, so cell size stays constant on
  // screen regardless of zoom) and rendered as a single bubble per cell
  // showing the dominant mode's emoji + a stop count; clicking a bubble
  // zooms/pans to fit its stops. Only the *exact* selected stops (a clicked
  // stop's own parent-station group, or -- in the trip box -- that trip's
  // own stops) are exempted from clustering and always shown individually;
  // every other stop, including the rest of a highlighted route's stops
  // (aboveLineIds), still clusters normally at low zoom regardless of
  // whether a box happens to be open.
  // ------------------------------------------------------------------
  var CLUSTER_ZOOM = 15;
  var CLUSTER_CELL_PX = 56;
  var clusterLayer = L.layerGroup().addTo(map);

  function updateMarkers() {
    var zoom = map.getZoom();
    var clustering = zoom < CLUSTER_ZOOM;

    var exempt = {};
    if (lastHighlight) {
      (lastHighlight.selectedIds || []).forEach(function (id) { exempt[id] = true; });
    }

    if (!clustering) {
      clusterLayer.clearLayers();
      Object.keys(markers).forEach(function (sid) {
        var marker = markers[sid];
        if (stopVisible(sid)) {
          if (!map.hasLayer(marker)) marker.addTo(map);
        } else if (map.hasLayer(marker)) {
          map.removeLayer(marker);
        }
      });
      return;
    }

    Object.keys(markers).forEach(function (sid) {
      if (map.hasLayer(markers[sid])) map.removeLayer(markers[sid]);
    });
    clusterLayer.clearLayers();

    var cells = {};
    Object.keys(DATA.stops).forEach(function (sid) {
      if (!stopVisible(sid)) return;
      if (exempt[sid]) {
        markers[sid].addTo(map);
        return;
      }
      var stop = DATA.stops[sid];
      if (stop.lat === null || stop.lon === null) return;
      var pt = map.project([stop.lat, stop.lon], zoom);
      var key = Math.floor(pt.x / CLUSTER_CELL_PX) + "_" + Math.floor(pt.y / CLUSTER_CELL_PX);
      (cells[key] = cells[key] || []).push(sid);
    });

    Object.keys(cells).forEach(function (key) {
      var ids = cells[key];
      if (ids.length === 1) {
        var sid = ids[0];
        markers[sid].addTo(map);
        return;
      }

      var bestMode = null;
      var bestRank = -Infinity;
      var sumLat = 0;
      var sumLon = 0;
      ids.forEach(function (id) {
        var stop = DATA.stops[id];
        sumLat += stop.lat;
        sumLon += stop.lon;
        (stop.modes || []).forEach(function (t) {
          var r = MODE_Z_RANK.hasOwnProperty(t) ? MODE_Z_RANK[t] : MODE_Z_OTHER;
          if (r > bestRank) {
            bestRank = r;
            bestMode = t;
          }
        });
      });
      var emoji = (bestMode && (routeTypeEmoji[bestMode] || routeTypeEmojiFallback)) || routeTypeEmojiFallback;
      var icon = L.divIcon({
        html: '<div class="gtfs-rm-cluster">' + emoji + '<span class="gtfs-rm-cluster-count">' + ids.length + "</span></div>",
        className: "",
        iconSize: [34, 34],
        iconAnchor: [17, 17],
      });
      var clusterMarker = L.marker([sumLat / ids.length, sumLon / ids.length], { icon: icon });
      clusterMarker.on("click", function () {
        var bounds = L.latLngBounds(
          ids.map(function (id) {
            var s = DATA.stops[id];
            return [s.lat, s.lon];
          })
        );
        map.flyToBounds(bounds, { maxZoom: CLUSTER_ZOOM + 2, padding: [40, 40] });
      });
      clusterMarker.addTo(clusterLayer);
    });
  }

  // Kept as the name the rest of the file already calls on filter changes.
  function applyFilter() {
    updateMarkers();
  }

  map.on("zoomend moveend", updateMarkers);
  updateMarkers();

  // Remembers the args of the last highlightStops() call so mode-checkbox
  // changes can cheaply re-render marker badges/visibility without
  // recomputing them (see the checkbox change handler above).
  var lastHighlight = null;

  // selectedIds: exact stop_ids to show large/selected (thick outline).
  // aboveLineIds (optional, defaults to selectedIds): every stop_id that
  // belongs to a currently-highlighted route -- these stay z-order above
  // the drawn linestring(s) even if not individually "selected"; anything
  // else gets pushed below the line while a highlight is active.
  // badgesBySid (optional): { stop_id: [route_id, ...] } line badges to
  // render beside the emoji -- already sorted/excluded by the caller;
  // mode-filtering is (re-)applied here so it stays live on checkbox change.
  function highlightStops(selectedIds, aboveLineIds, badgesBySid) {
    lastHighlight = { selectedIds: selectedIds, aboveLineIds: aboveLineIds, badgesBySid: badgesBySid };
    var selSet = {};
    selectedIds.forEach(function (id) {
      selSet[id] = true;
    });
    var aboveSet = {};
    (aboveLineIds || selectedIds).forEach(function (id) {
      aboveSet[id] = true;
    });
    var anyActive = selectedIds.length > 0 || (aboveLineIds || []).length > 0;
    Object.keys(markers).forEach(function (sid) {
      var marker = markers[sid];
      var isSelected = !!selSet[sid];
      var isAbove = isSelected || !!aboveSet[sid];
      var badges = badgesBySid && badgesBySid[sid] ? visibleRoutes(badgesBySid[sid]) : null;
      marker.setIcon(makeIcon(DATA.stops[sid], isSelected, badges));
      var base = modeBaseOffset(sid);
      if (isAbove) {
        marker.setZIndexOffset(base + (isSelected ? 100000 : 50000));
        movePane(marker, "markerPane");
      } else if (anyActive) {
        marker.setZIndexOffset(base);
        movePane(marker, "gtfsBelowLinePane");
      } else {
        marker.setZIndexOffset(base);
        movePane(marker, "markerPane");
      }
    });
  }

  function clearShapeHighlight() {
    highlightLayer.clearLayers();
  }

  function drawShape(shapeId, color) {
    var pts = DATA.shapes[shapeId];
    if (!pts || !pts.length) return;
    L.polyline(pts, { color: "#" + color, weight: 4, opacity: 0.85, pane: "gtfsLinePane" }).addTo(highlightLayer);
  }

  // ------------------------------------------------------------------
  // Floating boxes
  // ------------------------------------------------------------------
  var boxContainer = document.createElement("div");
  boxContainer.className = "gtfs-rm";
  map.getContainer().appendChild(boxContainer);
  // Scrolling/wheeling inside the box (or any of its scrollable
  // sub-sections) must scroll that content, not zoom the map underneath.
  L.DomEvent.disableScrollPropagation(boxContainer);

  // Every route icon (marker badges, per-platform/connections badge grids,
  // the trip-box title badge, ...) is clickable and opens that route's own
  // trip timetable -- except badges explicitly marked non-openable
  // (data-openable="false": the timetable's own route-filter row and the
  // "Filter lines..." checkboxes, which already have their own click
  // behavior and shouldn't also navigate away).
  //
  // The map-container listener runs on the *capture* phase, ahead of
  // Leaflet's own bubble-phase marker click handler -- a badge lives inside
  // a marker's divIcon, and Leaflet's marker click listener otherwise stops
  // the event before it would ever reach a bubble-phase delegated listener.
  function handleBadgeOpenClick(ev) {
    var badge = ev.target.closest && ev.target.closest(".gtfs-rm-badge");
    if (!badge || badge.getAttribute("data-openable") === "false") return;
    var routeId = badge.getAttribute("data-route");
    if (!routeId) return;
    ev.stopPropagation();
    if (ev.preventDefault) ev.preventDefault();
    openRouteTrip(routeId);
  }
  map.getContainer().addEventListener("click", handleBadgeOpenClick, true);
  boxContainer.addEventListener("click", handleBadgeOpenClick);

  // Tracks which parent_station is currently the subject of the open stop
  // box, so a second click on any of its already-highlighted platforms
  // narrows the timetable down to that one stop_id instead of the merged
  // station view (see selectStop below).
  var selectedParent = null;

  function closeBox() {
    boxContainer.innerHTML = "";
  }

  // pinnedHtml (optional): content that stays fixed at the top of the box,
  // outside the scrollable body -- e.g. the route-icon filter row, which
  // must always stay visible regardless of how long the stop/timetable
  // lists below it get.
  function openBox(titleHtml, bodyHtml, navHtml, pinnedHtml) {
    boxContainer.innerHTML =
      '<div class="gtfs-rm-box">' +
      '<div class="gtfs-rm-box-header">' +
      (navHtml || "") +
      '<div class="gtfs-rm-box-title" style="flex:1">' + titleHtml + "</div>" +
      '<button class="gtfs-rm-box-close" id="gtfs-rm-close-btn">×</button>' +
      "</div>" +
      (pinnedHtml ? '<div class="gtfs-rm-box-pinned">' + pinnedHtml + "</div>" : "") +
      '<div class="gtfs-rm-box-body">' + bodyHtml + "</div>" +
      "</div>";
    document.getElementById("gtfs-rm-close-btn").addEventListener("click", function () {
      closeBox();
      clearShapeHighlight();
      highlightStops([]);
      selectedParent = null;
      updateMarkers();
    });
  }

  // ------------------------------------------------------------------
  // "Filter lines..." box: one checkbox per route (grouped by mode, each
  // group and each route within it ordered most- to least-served), plus
  // select-all/clear-all. Independent of, and combined with (AND), the
  // mode checkboxes in applyFilter().
  // ------------------------------------------------------------------
  function openRouteFilterBox() {
    var modesSorted = presentTypes.slice().sort(function (a, b) {
      var ra = MODE_Z_RANK.hasOwnProperty(a) ? MODE_Z_RANK[a] : MODE_Z_OTHER;
      var rb = MODE_Z_RANK.hasOwnProperty(b) ? MODE_Z_RANK[b] : MODE_Z_OTHER;
      return rb - ra;
    });

    var routesByMode = {};
    Object.keys(DATA.routes).forEach(function (rid) {
      var rt = DATA.routes[rid].route_type;
      (routesByMode[rt] = routesByMode[rt] || []).push(rid);
    });

    var listHtml = modesSorted
      .map(function (t) {
        var routes = sortByService(routesByMode[t] || []);
        if (!routes.length) return "";
        var emoji = routeTypeEmoji[t] || routeTypeEmojiFallback;
        var name = routeTypeName[t] || routeTypeNameFallback;
        return (
          '<div class="gtfs-rm-linefilter-group">' +
          '<div class="gtfs-rm-linefilter-mode">' + emoji + " " + escapeHtml(name) + "</div>" +
          routes
            .map(function (rid) {
              var r = DATA.routes[rid];
              var checked = checkedRoutes[rid] !== false;
              return (
                '<label class="gtfs-rm-linefilter-row">' +
                '<input type="checkbox" ' + (checked ? "checked" : "") + ' data-route="' + rid + '">' +
                badgeHtml(rid, null, null, false) +
                '<span class="gtfs-rm-linefilter-name">' + escapeHtml(r.route_long_name || "") + "</span>" +
                "</label>"
              );
            })
            .join("") +
          "</div>"
        );
      })
      .join("");

    var pinned =
      '<div class="gtfs-rm-linefilter-actions">' +
      '<button type="button" class="gtfs-rm-btn" id="gtfs-rm-lf-all">Select all</button>' +
      '<button type="button" class="gtfs-rm-btn gtfs-rm-btn-secondary" id="gtfs-rm-lf-none">Clear all</button>' +
      "</div>";

    var body = '<div class="gtfs-rm-box-tablewrap">' + listHtml + "</div>";

    openBox("Filter lines", body, null, pinned);

    function setAll(value) {
      Object.keys(DATA.routes).forEach(function (rid) {
        checkedRoutes[rid] = value;
      });
      boxContainer.querySelectorAll(".gtfs-rm-linefilter-row input[type=checkbox]").forEach(function (cb) {
        cb.checked = value;
      });
      applyFilter();
      refreshLiveHighlight();
    }

    boxContainer.querySelectorAll(".gtfs-rm-linefilter-row input[type=checkbox]").forEach(function (cb) {
      cb.addEventListener("change", function () {
        checkedRoutes[cb.getAttribute("data-route")] = cb.checked;
        applyFilter();
        refreshLiveHighlight();
      });
    });
    document.getElementById("gtfs-rm-lf-all").addEventListener("click", function () {
      setAll(true);
    });
    document.getElementById("gtfs-rm-lf-none").addEventListener("click", function () {
      setAll(false);
    });
  }

  // ------------------------------------------------------------------
  // Stop click -> whole-parent-station timetable, narrowing to a single
  // platform on a second click.
  //
  // Clicking any one platform (stop_id) highlights every stop_id sharing
  // its parent_station and opens one merged timetable box for the group,
  // plus (when the station has more than one platform) a per-platform
  // breakdown of which lines actually call at each one. Clicking one of
  // those already-highlighted platforms again narrows the timetable to
  // just that stop_id. The top line-badge row is clickable (Google-Maps
  // style) to filter the timetable to a single route.
  // ------------------------------------------------------------------
  function selectStop(sid) {
    var stop = DATA.stops[sid];
    if (!stop) return;
    var parent = stop.parent || sid;
    var groupIds = stopsByParent[parent] || [sid];

    // A second click on a stop_id that's already highlighted (i.e. we're
    // already showing this parent_station's group) narrows *everything* --
    // highlighting, badges and the timetable -- down to that one stop_id.
    var singleStopMode = selectedParent === parent;
    selectedParent = parent;
    var highlightSet = singleStopMode ? [sid] : groupIds;

    var routeSet = [];
    highlightSet.forEach(function (id) {
      (DATA.stop_routes[id] || []).forEach(function (r) {
        if (routeSet.indexOf(r) === -1) routeSet.push(r);
      });
    });
    routeSet = sortByService(routeSet);

    // Every stop served by any of those routes stays z-order above the
    // linestring(s) drawn below, even though only `highlightSet` gets the
    // large "selected" treatment.
    var aboveLineIds = {};
    routeSet.forEach(function (r) {
      (routeStops[r] || []).forEach(function (id) {
        aboveLineIds[id] = true;
      });
    });
    highlightSet.forEach(function (id) {
      aboveLineIds[id] = true;
    });
    aboveLineIds = Object.keys(aboveLineIds);

    var badgesBySid = {};
    highlightSet.forEach(function (id) {
      badgesBySid[id] = sortByService(DATA.stop_routes[id] || []);
    });
    highlightStops(highlightSet, aboveLineIds, badgesBySid);
    updateMarkers();
    clearShapeHighlight();

    // How many times each route actually stops here (in the highlighted
    // group) -- used to z-order the drawn linestrings among themselves:
    // the more a route serves this stop, the higher its line sits.
    var stopCountByRoute = {};
    highlightSet.forEach(function (id) {
      (DATA.departures[id] || []).forEach(function (d) {
        stopCountByRoute[d.route_id] = (stopCountByRoute[d.route_id] || 0) + 1;
      });
    });

    var shapeSet = {};
    highlightSet.forEach(function (id) {
      (DATA.stop_shapes[id] || []).forEach(function (sh) {
        shapeSet[sh] = true;
      });
    });
    // Draw least-frequent-here route first, most-frequent last -- later
    // SVG paths paint on top, so this puts the busiest route's line
    // uppermost among the highlighted linestrings.
    Object.keys(shapeSet)
      .sort(function (a, b) {
        var ca = stopCountByRoute[DATA.shape_route[a]] || 0;
        var cb = stopCountByRoute[DATA.shape_route[b]] || 0;
        return ca - cb;
      })
      .forEach(function (shId) {
        var rId = DATA.shape_route[shId];
        var color = rId && DATA.routes[rId] ? DATA.routes[rId].route_color : "3388ff";
        drawShape(shId, color);
      });

    var departures = [];
    highlightSet.forEach(function (id) {
      (DATA.departures[id] || []).forEach(function (d) {
        // origin (which exact stop_id this departure is from) is needed to
        // look up that stop+route's own speed/headway below -- a merged
        // parent-station timetable can combine departures from several
        // platforms at once.
        departures.push(Object.assign({ origin: id }, d));
      });
    });
    departures.sort(function (a, b) {
      return (a.dep_time || 0) - (b.dep_time || 0);
    });

    function routeMetric(d) {
      var byStop = DATA.stop_route_metrics && DATA.stop_route_metrics[d.origin];
      return (byStop && byStop[d.route_id]) || null;
    }

    // Multiple routes can be selected at once (Google-Maps-style): an empty
    // set means "no filter", i.e. every route's departures show, and every
    // badge renders as checked -- ticking one or more badges narrows the
    // table (and the map highlight) down to just those routes' stops.
    var activeFilters = {};

    function filteredDepartures() {
      var keys = Object.keys(activeFilters);
      if (!keys.length) return departures;
      return departures.filter(function (d) {
        return activeFilters[d.route_id];
      });
    }

    function renderRows() {
      return filteredDepartures()
        .map(function (d) {
          var dest = DATA.stops[d.dest_stop_id] || {};
          var metric = routeMetric(d);
          return (
            '<tr class="gtfs-rm-row-clickable" data-trip="' + d.trip_id + '">' +
            "<td>" + badgeHtml(d.route_id) + "</td>" +
            "<td>" + escapeHtml(dest.stop_name || "") +
            (dest.parent_name && dest.parent_name !== dest.stop_name
              ? '<span class="gtfs-rm-dest-parent">' + escapeHtml(dest.parent_name) + "</span>"
              : "") +
            "</td>" +
            '<td class="gtfs-rm-time">' + fmtTime(d.dep_time) + "</td>" +
            '<td class="gtfs-rm-metric">' + fmtSpeed(metric && metric.speed) + "</td>" +
            '<td class="gtfs-rm-metric">' + fmtHeadwayMin(metric && metric.headway) + "</td>" +
            "</tr>"
          );
        })
        .join("");
    }

    // Reflects the average headway/speed of whatever's actually on screen
    // right now (filtered by the active route badge, if any) -- recomputed
    // alongside the table every time that filter changes. Average speed is
    // the mean of each visible departure's own (stop, route) speed, since
    // -- unlike headway -- it can't be derived from the departure times
    // alone.
    function updateHeadway() {
      var el = document.getElementById("gtfs-rm-headway");
      if (!el) return;
      var rows = filteredDepartures();
      var avgHeadway = computeAvgHeadwaySecs(rows);
      var speeds = rows
        .map(function (d) {
          var m = routeMetric(d);
          return m && m.speed;
        })
        .filter(function (v) {
          return v !== null && v !== undefined && !isNaN(v);
        });
      var avgSpeed = speeds.length ? speeds.reduce(function (a, b) { return a + b; }, 0) / speeds.length : null;
      el.innerHTML =
        "Avg. headway: " + fmtDuration(avgHeadway) + " &nbsp;·&nbsp; Avg. speed: " + fmtSpeed(avgSpeed);
    }

    // No badge toggled off yet means every route's departures are showing --
    // reflect that visually by starting every badge in the "active"
    // (checked-looking) state, not none of them.
    var badgeRow =
      '<div class="gtfs-rm-badge-row">' +
      routeSet
        .map(function (r) {
          return badgeHtml(r, "gtfs-rm-badge-filter active", null, false);
        })
        .join("") +
      "</div>" +
      '<div class="gtfs-rm-headway" id="gtfs-rm-headway"></div>';

    var platformsHtml = "";
    if (!singleStopMode && groupIds.length > 1) {
      var platformIds = groupIds.filter(function (id) {
        return (DATA.stop_routes[id] || []).length > 0;
      });
      if (platformIds.length) {
        platformsHtml =
          '<div class="gtfs-rm-box-platforms-scroll"><div class="gtfs-rm-platforms">' +
          platformIds
            .map(function (id) {
              var s = DATA.stops[id];
              return (
                '<div class="gtfs-rm-platform-row">' +
                '<div class="gtfs-rm-platform-name">' + escapeHtml(s.stop_name || id) + "</div>" +
                '<div class="gtfs-rm-badge-row">' +
                sortByService(DATA.stop_routes[id] || []).map(function (r) { return badgeHtml(r); }).join("") +
                "</div></div>"
              );
            })
            .join("") +
          "</div></div>";
      }
    }

    // The route-icon row stays pinned above the scrolling content; the
    // (optional) per-platform breakdown and the timetable each get their
    // own independent scroll area, so a long stop_id list never pushes the
    // timetable out of view.
    var body =
      platformsHtml +
      '<div class="gtfs-rm-box-tablewrap"><table class="gtfs-rm-table"><thead><tr><th>Route</th><th>Destination</th><th>Dep.</th><th>Speed</th><th>Headway</th></tr></thead>' +
      '<tbody id="gtfs-rm-dep-tbody">' + renderRows() + "</tbody></table></div>";

    var title = singleStopMode ? stop.stop_name || sid : stop.parent_name || stop.stop_name || sid;
    openBox(escapeHtml(title), body, null, badgeRow);
    updateHeadway();

    function wireRowClicks() {
      boxContainer.querySelectorAll("tr[data-trip]").forEach(function (tr) {
        tr.addEventListener("click", function () {
          openTripBox(tr.getAttribute("data-trip"));
        });
      });
    }
    wireRowClicks();

    // Route icons in this pinned row are *only* a timetable filter -- they
    // never change what's highlighted/elevated on the map (that stays
    // whatever selectStop() set up for the whole highlightSet/aboveLineIds
    // once, above). Toggle rules (all in terms of "active" = shown):
    //  - every route active (the default) + click one -> isolate it (only
    //    that one stays active).
    //  - exactly one route active + click that same one -> back to every
    //    route active.
    //  - otherwise -> plain toggle of that one route's membership, folding
    //    back to "every route active" (the canonical empty-set state) if
    //    that happens to leave literally all of them checked.
    function toggleRouteFilter(r) {
      var keys = Object.keys(activeFilters);
      var allActive = keys.length === 0 || keys.length === routeSet.length;
      if (allActive) {
        activeFilters = {};
        activeFilters[r] = true;
      } else if (keys.length === 1 && keys[0] === r) {
        activeFilters = {};
      } else if (activeFilters[r]) {
        delete activeFilters[r];
      } else {
        activeFilters[r] = true;
        if (Object.keys(activeFilters).length === routeSet.length) activeFilters = {};
      }
    }

    boxContainer.querySelectorAll(".gtfs-rm-badge-filter").forEach(function (b) {
      b.addEventListener("click", function (ev) {
        ev.stopPropagation();
        toggleRouteFilter(b.getAttribute("data-route"));

        var anyFilter = Object.keys(activeFilters).length > 0;
        // No filter active means every route's times are showing again --
        // show every badge as "checked" rather than none of them.
        boxContainer.querySelectorAll(".gtfs-rm-badge-filter").forEach(function (bb) {
          bb.classList.toggle("active", anyFilter ? !!activeFilters[bb.getAttribute("data-route")] : true);
        });
        document.getElementById("gtfs-rm-dep-tbody").innerHTML = renderRows();
        wireRowClicks();
        updateHeadway();
      });
    });
  }

  // ------------------------------------------------------------------
  // Lines serving the parent_station of a given stop_id, memoized --
  // used for the per-stop 3x3 badge grid (station-wide connections) in
  // the trip-itinerary box.
  // ------------------------------------------------------------------
  var parentRoutesCache = {};
  function routesForParentOfStop(stopId) {
    var stop = DATA.stops[stopId];
    var parent = stop ? stop.parent || stopId : stopId;
    if (parentRoutesCache[parent]) return parentRoutesCache[parent];
    var ids = stopsByParent[parent] || [stopId];
    var set = [];
    ids.forEach(function (id) {
      (DATA.stop_routes[id] || []).forEach(function (r) {
        if (set.indexOf(r) === -1) set.push(r);
      });
    });
    set = sortByService(set);
    parentRoutesCache[parent] = set;
    return set;
  }

  // excludeRouteId: leave out the route currently being highlighted -- the
  // user already knows that line stops here, it's the reason this box/badge
  // is showing in the first place.
  function miniBadgeGrid(stopId, excludeRouteId) {
    var routes = visibleRoutes(excludeRoute(routesForParentOfStop(stopId), excludeRouteId));
    if (!routes.length) return "";
    return (
      '<div class="gtfs-rm-badge-grid-mini">' +
      routes.map(function (r) { return badgeHtml(r); }).join("") +
      "</div>"
    );
  }

  // ------------------------------------------------------------------
  // Timetable row click -> full trip itinerary box.
  //
  // Only the trip's own exact stop_ids are highlighted on the map (not
  // their whole parent_station groups); the per-stop connections grid
  // below is station-wide (routesForParentOfStop), so it shows everything
  // reachable there even though only that one platform is highlighted.
  // ------------------------------------------------------------------
  function openTripBox(tripId) {
    var trip = DATA.trips[tripId];
    if (!trip) return;

    clearShapeHighlight();
    if (trip.shape_id) drawShape(trip.shape_id, DATA.routes[trip.route_id] ? DATA.routes[trip.route_id].route_color : "3388ff");

    var tripStopIds = trip.stops.map(function (s) {
      return s.stop_id;
    });
    // Every stop the route serves stays z-order above the highlighted
    // linestring (only this trip's own stops get the large "selected"
    // treatment), badged with its *parent station's* other route
    // connections -- the highlighted route itself is left out, since it's
    // already obvious from the highlight.
    var routeStopIds = routeStops[trip.route_id] || tripStopIds;
    var badgesBySid = {};
    routeStopIds.forEach(function (sid) {
      badgesBySid[sid] = excludeRoute(routesForParentOfStop(sid), trip.route_id);
    });
    highlightStops(tripStopIds, routeStopIds.concat(tripStopIds), badgesBySid);
    updateMarkers();

    var rows = trip.stops
      .map(function (s) {
        var stop = DATA.stops[s.stop_id] || {};
        return (
          "<tr>" +
          '<td><span class="gtfs-rm-stop-link" data-stop="' + s.stop_id + '">' + escapeHtml(stop.stop_name || "") + "</span>" +
          (stop.parent_name && stop.parent_name !== stop.stop_name
            ? '<span class="gtfs-rm-dest-parent">' + escapeHtml(stop.parent_name) + "</span>"
            : "") +
          "</td>" +
          "<td>" + miniBadgeGrid(s.stop_id, trip.route_id) + "</td>" +
          '<td class="gtfs-rm-time">' + fmtTime(s.arr) + "</td>" +
          '<td class="gtfs-rm-time">' + fmtTime(s.dep) + "</td>" +
          "</tr>"
        );
      })
      .join("");

    var body =
      '<div class="gtfs-rm-box-tablewrap"><table class="gtfs-rm-table"><thead><tr><th>Stop</th><th>Lines</th><th>Arr.</th><th>Dep.</th></tr></thead><tbody>' +
      rows +
      "</tbody></table></div>";

    var navKey = trip.route_id + "||" + trip.direction_id;
    var sameRouteTrips = DATA.route_trips[navKey] || [tripId];
    var idx = sameRouteTrips.indexOf(tripId);
    var nav =
      '<button class="gtfs-rm-nav-btn" id="gtfs-rm-prev-btn">←</button>' +
      '<button class="gtfs-rm-nav-btn" id="gtfs-rm-next-btn">→</button>';

    var route = DATA.routes[trip.route_id] || {};
    var title = badgeHtml(trip.route_id) + " " + escapeHtml(trip.headsign || route.route_long_name || "");

    openBox(title, body, nav);

    boxContainer.querySelectorAll(".gtfs-rm-stop-link").forEach(function (el) {
      el.addEventListener("click", function (ev) {
        ev.stopPropagation();
        var stop = DATA.stops[el.getAttribute("data-stop")];
        if (stop && stop.lat !== null && stop.lon !== null) {
          map.flyTo([stop.lat, stop.lon], Math.max(map.getZoom(), 17));
        }
      });
    });

    if (sameRouteTrips.length > 1) {
      document.getElementById("gtfs-rm-prev-btn").addEventListener("click", function () {
        var prevIdx = (idx - 1 + sameRouteTrips.length) % sameRouteTrips.length;
        openTripBox(sameRouteTrips[prevIdx]);
      });
      document.getElementById("gtfs-rm-next-btn").addEventListener("click", function () {
        var nextIdx = (idx + 1) % sameRouteTrips.length;
        openTripBox(sameRouteTrips[nextIdx]);
      });
    } else {
      document.getElementById("gtfs-rm-prev-btn").disabled = true;
      document.getElementById("gtfs-rm-next-btn").disabled = true;
    }
  }
})();
