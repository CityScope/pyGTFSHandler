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

  function numberedIcon(n, color, conflict) {
    var star = conflict ? '<div class="conflict-star-badge">★</div>' : "";
    return L.divIcon({
      className: "",
      html: '<div class="conflict-num-badge" style="background:' + color + '">' + n + "</div>" + star,
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
    if (s.conflict) {
      html +=
        '<span style="color:#d62728">★ conflict -- this stop\'s own geometry actually ' +
        "indicated direction_id <b>" + fmt(s.real_direction_id) + "</b>, not " + fmt(s.direction_id) +
        "</span><br>";
    }
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
      var tooltip = i + 1 + ": " + s.stop_id + (s.conflict ? " ★" : "");
      L.marker([lat, lon], { icon: numberedIcon(i + 1, color, s.conflict) })
        .bindTooltip(tooltip)
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
      html += "<u>By direction_id, longest shape_id first</u><br>";
      // Only one direction_id's shape can be shown in blue/red-reference
      // at a time (a single `okLayer`), so both dropdowns start on "None"
      // and picking one resets the other back to it -- see
      // `selectOkIndex`.
      dirKeys.forEach(function (dirKey) {
        var list = info.ok[dirKey];
        html += '<div style="margin-top:6px"><b>direction_id ' + dirKey + "</b> (" + list.length + ")<br>";
        html += '<select class="conflict-ok-select" id="conflict-dir-select-' +
          dirKey +
          '" onchange="window.__selectOkIndex(\'' +
          stationId +
          "','" +
          routeId +
          "','" +
          dirKey +
          "', this.value)\">";
        html += '<option value="-1">None</option>';
        list.forEach(function (entry, i) {
          var label =
            entry.shape_id + " (" + entry.length + "m, " + entry.stops.length + " stops)" +
            (entry.conflict ? " -- direction_conflict" : "");
          html +=
            '<option value="' + i + '"' + (entry.conflict ? ' style="color:#d62728"' : "") + ">" + label + "</option>";
        });
        html += "</select></div>";
      });
    } else {
      html += "<i>No shape_id for this route at this station.</i>";
    }

    shapesPanel.innerHTML = html;
    shapesPanel.style.display = "block";
    // Both dropdowns default to "None" -- nothing drawn until the user
    // picks a direction.
  }

  function selectOkIndex(stationId, routeId, dirKey, idx) {
    var info = DATA.conflicts[stationId][routeId];

    // Only one non-conflicting/reference shape (one `okLayer`) can be on
    // the map at a time. Picking a shape in one direction's dropdown
    // resets every *other* direction's dropdown back to "None", so the
    // two can never both claim to be the one shown.
    Object.keys(info.ok || {}).forEach(function (otherDirKey) {
      if (otherDirKey === dirKey) return;
      var otherSelect = document.getElementById("conflict-dir-select-" + otherDirKey);
      if (otherSelect) otherSelect.value = "-1";
    });

    if (okLayer) {
      map.removeLayer(okLayer);
      okLayer = null;
      okCoords = [];
    }

    var entry = info.ok[dirKey][idx];
    if (!entry) return; // idx === "-1" ("None"), or invalid -- leave cleared.

    okLayer = drawSequence(entry.stops, entry.conflict ? COLOR_CONFLICT : COLOR_OK);
    okCoords = entry.stops.map(function (s) {
      return [s.lat, s.lon];
    });
    // Redraw the conflicting layer (if any) on top, so its markers stay
    // above -- and stay correctly offset against -- the newly-selected
    // sequence.
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
      '"></span>Selected conflicting shape_id</div>' +
      '<div><span style="color:' +
      COLOR_CONFLICT +
      '">★</span> Stop where the geometry disagreed with the shape\'s reported direction_id</div>';
    return div;
  };
  legend.addTo(map);
})();
