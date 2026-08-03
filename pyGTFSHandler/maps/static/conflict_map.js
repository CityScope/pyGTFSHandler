(function waitForMap() {
  if (!window["__MAP_VAR__"]) {
    setTimeout(waitForMap, 25);
    return;
  }
  var map = window["__MAP_VAR__"];
  var DATA = __DATA_JSON__;

  var routesPanel = document.getElementById("conflict-panel-routes");
  var shapesPanel = document.getElementById("conflict-panel-shapes");

  // Two independent drawn sequences at a time: the route's non-conflicting
  // reference shape (blue) and whichever conflicting shape_id was last
  // clicked (red). Each is a LayerGroup (polyline + one numbered marker
  // per stop, in visit order) so it can be cleared and redrawn as a unit.
  var COLOR_CONFLICT = "#d62728";
  var COLOR_OK = "#1f6fd6";
  // A stop shared by both the blue and red sequences sits at the exact
  // same lat/lon in both, so its two numbered badges would otherwise be
  // drawn one on top of the other, and only the topmost stays clickable/
  // visible. Nudging the second marker drawn there by a few meters keeps
  // both fully visible and distinguishable without shifting the actual
  // (accurate) polyline geometry, which is drawn through the true,
  // unshifted coordinates.
  var COORD_MATCH_EPS = 1e-6;
  var OVERLAP_OFFSET = 0.00006; // ~6-7m at this latitude

  var okLayer = null;
  var conflictLayer = null;
  var okCoords = [];

  function numberedIcon(n, color) {
    return L.divIcon({
      className: "",
      html: '<div class="conflict-num-badge" style="background:' + color + '">' + n + "</div>",
      iconSize: [22, 22],
      iconAnchor: [11, 11],
    });
  }

  function coordMatches(coords, lat, lon) {
    for (var i = 0; i < coords.length; i++) {
      if (Math.abs(coords[i][0] - lat) < COORD_MATCH_EPS && Math.abs(coords[i][1] - lon) < COORD_MATCH_EPS) {
        return true;
      }
    }
    return false;
  }

  function fmt(v, suffix) {
    return v === null || v === undefined ? "n/a" : v + (suffix || "");
  }

  function stopPopupHtml(s) {
    var html = "<b>" + s.stop_id + "</b> (#" + s.__seq + ")<br>";
    html += "direction_id: <b>" + fmt(s.direction_id) + "</b><br>";
    if (s.split_angle === null) {
      html += "split: n/a (nothing to split at this stop)<br>";
    } else {
      var lo = s.split_angle;
      var mid = (s.split_angle + 180) % 360;
      html += "direction 0 range: " + fmt(Math.round(lo * 10) / 10, "°") + " – " + fmt(Math.round(mid * 10) / 10, "°") + "<br>";
      html += "direction 1 range: " + fmt(Math.round(mid * 10) / 10, "°") + " – " + fmt(Math.round(lo * 10) / 10, "°") + "<br>";
    }
    html += "forward: " + fmt(s.fwd_raw, "°") + " (raw) → " + fmt(s.fwd_corrected, "°") + " (forced-180)<br>";
    html += "backward: " + fmt(s.bwd_raw, "°") + " (raw) → " + fmt(s.bwd_corrected, "°") + " (forced-180)";
    return html;
  }

  function drawSequence(stops, color, avoidCoords) {
    var group = L.layerGroup();
    var latlngs = stops.map(function (s) {
      return [s.lat, s.lon];
    });
    L.polyline(latlngs, { color: color, weight: 3, opacity: 0.7 }).addTo(group);
    stops.forEach(function (s, i) {
      var lat = s.lat;
      var lon = s.lon;
      if (avoidCoords && coordMatches(avoidCoords, lat, lon)) {
        lat += OVERLAP_OFFSET;
        lon += OVERLAP_OFFSET;
      }
      s.__seq = i + 1;
      L.marker([lat, lon], { icon: numberedIcon(i + 1, color) })
        .bindTooltip(i + 1 + ": " + s.stop_id)
        .bindPopup(stopPopupHtml(s))
        .addTo(group);
    });
    group.addTo(map);
    return group;
  }

  function clearSequences() {
    if (okLayer) {
      map.removeLayer(okLayer);
      okLayer = null;
    }
    if (conflictLayer) {
      map.removeLayer(conflictLayer);
      conflictLayer = null;
    }
    okCoords = [];
    window.__lastConflictShape = null;
  }

  function closePanels() {
    routesPanel.style.display = "none";
    shapesPanel.style.display = "none";
    clearSequences();
  }

  function showRoutes(stationId) {
    var routeData = DATA.conflicts[stationId];
    if (!routeData) return;
    clearSequences();
    var html = '<span class="conflict-close" onclick="window.__closeConflictPanels()">&times;</span>';
    html += "<b>Station " + stationId + "</b><br>Conflicting on routes:<br>";
    Object.keys(routeData).forEach(function (routeId) {
      html +=
        '<button class="conflict-route-btn" onclick="window.__showConflictShapes(\'' +
        stationId +
        "','" +
        routeId +
        "')\">" +
        routeId +
        "</button>";
    });
    routesPanel.innerHTML = html;
    routesPanel.style.display = "block";
    shapesPanel.style.display = "none";
  }

  function showShapes(stationId, routeId) {
    var info = DATA.conflicts[stationId][routeId];
    clearSequences();

    var html = '<span class="conflict-close" onclick="window.__closeConflictPanels()">&times;</span>';
    html += "<b>Station " + stationId + " / Route " + routeId + "</b><br><br>";

    var shapeIds = Object.keys(info.conflicting);
    html += "<u>Conflicting shape_ids (" + shapeIds.length + ", click to draw in red)</u>";
    html += '<div class="conflict-shape-list">';
    shapeIds.forEach(function (sid) {
      html +=
        '<button class="conflict-shape-btn" onclick="window.__drawConflictShape(\'' +
        stationId +
        "','" +
        routeId +
        "','" +
        sid +
        "')\">" +
        sid +
        "</button>";
    });
    html += "</div><br>";

    var dirKeys = Object.keys(info.ok || {});
    if (dirKeys.length) {
      html += "<u>Non-conflicting shape_ids (blue), by direction_id</u><br>";
      dirKeys.forEach(function (dirKey) {
        html +=
          '<button class="conflict-route-btn" id="conflict-dir-btn-' +
          dirKey +
          '" onclick="window.__selectOkDirection(\'' +
          stationId +
          "','" +
          routeId +
          "','" +
          dirKey +
          '\')">direction_id ' +
          dirKey +
          " (" +
          info.ok[dirKey].length +
          ")</button>";
      });
      html += '<div id="conflict-ok-slider-box"></div>';
    } else {
      html += "<i>No non-conflicting shape_id for this route at this station.</i>";
    }

    shapesPanel.innerHTML = html;
    shapesPanel.style.display = "block";
  }

  function selectOkDirection(stationId, routeId, dirKey) {
    var info = DATA.conflicts[stationId][routeId];
    var list = info.ok[dirKey];
    if (!list || !list.length) return;

    document.querySelectorAll(".conflict-route-btn").forEach(function (b) {
      b.classList.remove("active");
    });
    var btn = document.getElementById("conflict-dir-btn-" + dirKey);
    if (btn) btn.classList.add("active");

    var box = document.getElementById("conflict-ok-slider-box");
    box.innerHTML =
      '<input type="range" min="0" max="' +
      (list.length - 1) +
      '" value="0" step="1" class="conflict-ok-slider" ' +
      'oninput="window.__selectOkIndex(\'' +
      stationId +
      "','" +
      routeId +
      "','" +
      dirKey +
      '\', this.value)"><br>' +
      '<span id="conflict-ok-slider-label"></span>';

    selectOkIndex(stationId, routeId, dirKey, 0);
  }

  function selectOkIndex(stationId, routeId, dirKey, idx) {
    var info = DATA.conflicts[stationId][routeId];
    var entry = info.ok[dirKey][idx];
    if (!entry) return;

    var label = document.getElementById("conflict-ok-slider-label");
    if (label) {
      label.textContent =
        (parseInt(idx, 10) + 1) + " / " + info.ok[dirKey].length + ": " + entry.shape_id +
        " (" + entry.length + "m, " + entry.stops.length + " stops)";
    }

    if (okLayer) {
      map.removeLayer(okLayer);
      okLayer = null;
    }
    okLayer = drawSequence(entry.stops, COLOR_OK);
    okCoords = entry.stops.map(function (s) {
      return [s.lat, s.lon];
    });
    // Redraw the conflicting layer (if any) on top, so its markers stay
    // above -- and stay correctly offset against -- the newly-selected
    // non-conflicting sequence.
    if (window.__lastConflictShape) {
      drawConflictShape(window.__lastConflictShape.stationId, window.__lastConflictShape.routeId, window.__lastConflictShape.shapeId);
    }
  }

  function drawConflictShape(stationId, routeId, shapeId) {
    var info = DATA.conflicts[stationId][routeId];
    var shape = info.conflicting[shapeId];
    if (!shape) return;
    if (conflictLayer) {
      map.removeLayer(conflictLayer);
      conflictLayer = null;
    }
    window.__lastConflictShape = { stationId: stationId, routeId: routeId, shapeId: shapeId };
    conflictLayer = drawSequence(shape.stops, COLOR_CONFLICT, okCoords);
  }

  window.__closeConflictPanels = closePanels;
  window.__showConflictShapes = showShapes;
  window.__selectOkDirection = selectOkDirection;
  window.__selectOkIndex = selectOkIndex;
  window.__drawConflictShape = drawConflictShape;

  DATA.stations.forEach(function (s) {
    var marker = L.circleMarker([s.lat, s.lon], {
      radius: s.conflict ? 6 : 3,
      color: s.conflict ? COLOR_CONFLICT : "#888888",
      fillColor: s.conflict ? COLOR_CONFLICT : "#888888",
      fillOpacity: s.conflict ? 0.9 : 0.35,
      weight: 1,
    }).addTo(map);
    if (s.conflict) {
      marker.on("click", function () {
        showRoutes(s.station);
      });
    }
  });

  var legend = L.control({ position: "bottomleft" });
  legend.onAdd = function () {
    var div = L.DomUtil.create("div", "conflict-legend");
    div.innerHTML =
      '<div><span class="conflict-legend-dot" style="background:' +
      COLOR_CONFLICT +
      '"></span>Conflicting station</div>' +
      '<div><span class="conflict-legend-dot" style="background:#888888"></span>Non-conflicting station</div>' +
      '<div><span class="conflict-legend-line" style="background:' +
      COLOR_OK +
      '"></span>Longest non-conflicting shape_id</div>' +
      '<div><span class="conflict-legend-line" style="background:' +
      COLOR_CONFLICT +
      '"></span>Selected conflicting shape_id</div>';
    return div;
  };
  legend.addTo(map);
})();
