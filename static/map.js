// Field Outlook page: pick a district or click the map, then show weather, soil,
// disease-risk outlook and field tips from /api/outlook.
(function () {
  const t = JSON.parse(document.getElementById("i18n").textContent);
  const districts = JSON.parse(document.getElementById("districts").textContent);
  const lang = document.documentElement.lang || "en";
  const select = document.getElementById("district-select");
  const $ = (id) => document.getElementById(id);

  const MAHARASHTRA_CENTER = [19.3, 76.4];
  let map = null;
  let pin = null;

  // The map needs Leaflet from the CDN; without internet the dropdown still works.
  if (window.L) {
    map = L.map("map", { scrollWheelZoom: false }).setView(MAHARASHTRA_CENTER, 6);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 12,
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(map);
    districts.forEach((d) => {
      L.circleMarker([d.lat, d.lon], { radius: 6, color: "#2f7d3b", weight: 2, fillColor: "#8bc34a", fillOpacity: 0.9 })
        .addTo(map)
        .bindTooltip(d.name)
        .on("click", () => choose(d.lat, d.lon, d.slug));
    });
    map.on("click", (e) => choose(e.latlng.lat, e.latlng.lng, ""));
  } else {
    $("map").classList.add("map-offline");
    $("map").textContent = t.map_unavailable;
  }

  select.addEventListener("change", () => {
    const opt = select.selectedOptions[0];
    if (opt && opt.value) choose(parseFloat(opt.dataset.lat), parseFloat(opt.dataset.lon), opt.value);
  });

  function show(state) {
    $("outlook-placeholder").hidden = state !== "idle";
    $("outlook-loading").hidden = state !== "loading";
    $("outlook").hidden = state !== "ready";
    if (state !== "error") $("outlook-error").hidden = true;
  }

  async function choose(lat, lon, slug) {
    select.value = slug;
    if (map) {
      if (pin) pin.setLatLng([lat, lon]);
      else pin = L.marker([lat, lon]).addTo(map);
      map.panTo([lat, lon]);
    }
    show("loading");
    try {
      const params = new URLSearchParams({ lat: lat.toFixed(4), lon: lon.toFixed(4), lang });
      const response = await fetch("/api/outlook?" + params.toString());
      const data = await response.json();
      if (!response.ok) {
        show("idle");
        $("outlook-error").textContent = data.error || t.error_generic;
        $("outlook-error").hidden = false;
        return;
      }
      render(data);
      show("ready");
    } catch (err) {
      show("idle");
      $("outlook-error").textContent = t.error_network;
      $("outlook-error").hidden = false;
    }
  }

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  const fmt = (v, unit) => (v === null || v === undefined ? "–" : v + unit);

  function dayCard(day) {
    const card = el("div", "day");
    const date = new Date(day.date + "T00:00:00");
    card.append(
      el("strong", "day-name", date.toLocaleDateString(lang === "en" ? "en-IN" : lang + "-IN", { weekday: "short", day: "numeric" })),
      el("span", "day-temp", fmt(day.t_max, "°") + " / " + fmt(day.t_min, "°")),
      el("span", "day-meta", "💧 " + fmt(day.humidity_mean, "%")),
      el("span", "day-meta", "🌧 " + fmt(day.rain_mm, " mm")),
      el("span", "day-meta", "💨 " + fmt(day.wind_max, " km/h"))
    );
    if ((day.rain_mm || 0) >= 20) card.classList.add("day-wet");
    return card;
  }

  function render(data) {
    $("o-place").textContent = data.district.name;
    $("o-coords").textContent = data.lat.toFixed(3) + ", " + data.lon.toFixed(3);
    $("o-offline").hidden = data.weather.source !== "offline";
    $("o-days").replaceChildren(...data.weather.days.map(dayCard));

    $("o-risks").replaceChildren(
      ...data.risks.map((r) => {
        const li = el("li", "risk risk-" + r.level);
        const head = el("div", "risk-head");
        head.append(el("strong", null, r.crop_name + " · " + r.disease_name), el("span", "severity severity-" + r.level_class, r.level_text));
        li.append(head, el("p", "muted", r.reason_text + (r.peak_text ? " " + r.peak_text : "")));
        return li;
      })
    );

    const soil = data.soil;
    $("o-soil-type").textContent = soil.type_name || "–";
    $("o-soil-desc").textContent = soil.type_description || "";
    const props = [
      [t.soil_texture, soil.texture_name],
      [t.soil_ph, soil.ph],
      [t.soil_clay, soil.clay_pct !== null && soil.clay_pct !== undefined ? soil.clay_pct + "%" : null],
      [t.soil_sand, soil.sand_pct !== null && soil.sand_pct !== undefined ? soil.sand_pct + "%" : null],
      [t.soil_silt, soil.silt_pct !== null && soil.silt_pct !== undefined ? soil.silt_pct + "%" : null],
      [t.soil_oc, soil.organic_carbon_gkg !== null && soil.organic_carbon_gkg !== undefined ? soil.organic_carbon_gkg + " g/kg" : null],
    ].filter(([, v]) => v !== null && v !== undefined && v !== "");
    $("o-soil-props").replaceChildren(...props.flatMap(([k, v]) => [el("dt", null, k), el("dd", null, String(v))]));
    $("o-soil-source").textContent = soil.source === "live" ? t.soil_source_live : t.soil_source_district;

    $("o-tips").replaceChildren(...data.tips.map((tip) => el("li", null, tip.text)));
  }

  show("idle");
})();
