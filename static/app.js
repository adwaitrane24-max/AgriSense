// Loaded on every page.
// 1. Language: remember the chosen language and reopen pages in it.
// 2. Upload page only: preview the image, POST it to /predict, render the result.
// All text comes from the strings embedded by the server in <script id="i18n">.
// Server text is always inserted with textContent, never innerHTML.
(function () {
  const SUPPORTED = ["en", "hi", "mr"];
  const STORAGE_KEY = "leafcheck-lang";
  const lang = document.documentElement.lang || "en";
  const t = JSON.parse(document.getElementById("i18n").textContent);

  // localStorage can throw (private mode, blocked storage); the page must still work.
  function loadLang() {
    try {
      return localStorage.getItem(STORAGE_KEY);
    } catch (err) {
      return null;
    }
  }

  function saveLang(code) {
    try {
      localStorage.setItem(STORAGE_KEY, code);
    } catch (err) {
      /* ignore: the choice just will not be remembered */
    }
  }

  const params = new URLSearchParams(window.location.search);
  if (params.has("lang")) {
    saveLang(lang); // the server already normalised it
  } else {
    const saved = loadLang();
    if (saved && saved !== lang && SUPPORTED.includes(saved)) {
      params.set("lang", saved); // URL now has ?lang=, so this happens at most once
      window.location.replace(window.location.pathname + "?" + params.toString());
      return;
    }
  }
  document.querySelectorAll(".lang-switch a").forEach((link) =>
    link.addEventListener("click", () => saveLang(link.dataset.lang))
  );

  const form = document.getElementById("upload-form");
  if (!form) return; // not the upload page

  const input = document.getElementById("file-input");
  const dropzone = document.getElementById("dropzone");
  const preview = document.getElementById("preview");
  const dropText = document.getElementById("drop-text");
  const button = document.getElementById("submit-btn");
  const errorBox = document.getElementById("error");
  const placeholder = document.getElementById("placeholder");
  const resultBox = document.getElementById("result");
  const SEVERITIES = ["none", "low", "medium", "high"];

  let selectedFile = null;

  function choose(file) {
    selectedFile = file || null;
    errorBox.hidden = true;
    button.disabled = !selectedFile;
    if (!selectedFile) return;
    preview.src = URL.createObjectURL(selectedFile);
    preview.hidden = false;
    dropText.hidden = true;
  }

  input.addEventListener("change", () => choose(input.files[0]));

  ["dragenter", "dragover"].forEach((name) =>
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragging");
    })
  );
  ["dragleave", "drop"].forEach((name) =>
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragging");
    })
  );
  dropzone.addEventListener("drop", (event) => choose(event.dataTransfer.files[0]));

  function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
  }

  function percent(value) {
    return (value * 100).toFixed(1) + "%";
  }

  // Translated names when the server sends them, otherwise the English ones.
  function cropName(item) {
    return item.display_crop || item.crop;
  }

  function conditionName(item) {
    return item.display_condition || item.condition;
  }

  function renderAdvisory(advisory) {
    const card = document.getElementById("advisory");
    if (!advisory) {
      card.hidden = true;
      return;
    }
    const level = SEVERITIES.includes(advisory.severity) ? advisory.severity : "none";
    const badge = document.getElementById("advisory-severity");
    badge.className = "severity severity-" + level;
    badge.textContent = t.severity_label + ": " + t["severity_" + level];
    document.getElementById("advisory-harm").textContent = advisory.harm || "";
    document.getElementById("advisory-economic").textContent = advisory.economic_risk || "";
    document.getElementById("advisory-steps").replaceChildren(
      ...(advisory.next_steps || []).map((step) => {
        const li = document.createElement("li");
        li.textContent = step;
        return li;
      })
    );
    card.hidden = false;
  }

  function topItem(item) {
    const li = document.createElement("li");
    const row = document.createElement("div");
    row.className = "top-row";
    const name = document.createElement("span");
    name.textContent = cropName(item) + " — " + conditionName(item);
    const score = document.createElement("strong");
    score.textContent = percent(item.confidence);
    row.append(name, score);

    const bar = document.createElement("div");
    bar.className = "bar";
    const fill = document.createElement("div");
    fill.className = "bar-fill " + (item.healthy ? "bar-hit" : "bar-miss");
    fill.style.width = percent(item.confidence);
    bar.append(fill);

    li.append(row, bar);
    return li;
  }

  const WARNINGS = ["too_dark", "too_bright", "blurry", "not_leaf"];
  const BANDS = ["none", "mild", "moderate", "severe"];

  // Heatmap, affected-area meter and photo-quality warnings (all optional fields).
  function renderInsights(data) {
    const heatmapBox = document.getElementById("heatmap-box");
    const affectedBox = document.getElementById("affected-box");
    const leaf = data.leaf_analysis;

    heatmapBox.hidden = !data.heatmap;
    if (data.heatmap) document.getElementById("heatmap-img").src = data.heatmap;

    const showAffected = !!leaf && !(leaf.warnings || []).includes("not_leaf");
    affectedBox.hidden = !showAffected;
    if (showAffected) {
      const band = BANDS.includes(leaf.severity_band) ? leaf.severity_band : "none";
      document.getElementById("affected-pct").textContent = "≈ " + leaf.affected_percent.toFixed(1) + "%";
      const bandEl = document.getElementById("affected-band");
      bandEl.textContent = t["band_" + band];
      bandEl.className = "band band-" + band;
      document.getElementById("affected-fill").style.width = Math.min(100, leaf.affected_percent) + "%";
    }
    document.getElementById("insights").hidden = heatmapBox.hidden && affectedBox.hidden;

    const warnBox = document.getElementById("quality-warnings");
    const warnings = ((leaf && leaf.warnings) || []).filter((w) => WARNINGS.includes(w));
    warnBox.replaceChildren(
      ...warnings.map((w) => {
        const li = document.createElement("li");
        li.textContent = "⚠ " + t["warn_" + w];
        return li;
      })
    );
    warnBox.hidden = warnings.length === 0;
  }

  // ---- Voice readout (browser speech synthesis, no server or library needed)
  const VOICE_LANGS = { en: "en-IN", hi: "hi-IN", mr: "mr-IN" };
  const listenBtn = document.getElementById("listen-btn");
  let lastResult = null;

  function speechText(data) {
    if (data.status !== "ok") return t.not_sure_message + " " + t.disclaimer;
    const p = data.prediction;
    const parts = [
      cropName(p) + ", " + conditionName(p) + ". " + t.voice_confidence + " " + Math.round(p.confidence * 100) + "%.",
    ];
    if (data.advisory) {
      parts.push(t.advisory_harm + ": " + data.advisory.harm);
      parts.push(t.advisory_economic + ": " + data.advisory.economic_risk);
      parts.push(t.advisory_next_steps + ": " + data.advisory.next_steps.join(" "));
    }
    parts.push(t.disclaimer);
    return parts.join(" ");
  }

  function setListening(on) {
    listenBtn.textContent = (on ? "⏹ " + t.stop_listening : "🔊 " + t.listen);
  }

  listenBtn.addEventListener("click", () => {
    if (!("speechSynthesis" in window)) {
      showError(t.voice_unavailable);
      return;
    }
    if (speechSynthesis.speaking) {
      speechSynthesis.cancel();
      setListening(false);
      return;
    }
    if (!lastResult) return;
    const utterance = new SpeechSynthesisUtterance(speechText(lastResult));
    utterance.lang = VOICE_LANGS[lang] || "en-IN";
    const voice = speechSynthesis.getVoices().find((v) => v.lang === utterance.lang)
      || speechSynthesis.getVoices().find((v) => v.lang.startsWith(lang));
    if (voice) utterance.voice = voice;
    utterance.onend = () => setListening(false);
    setListening(true);
    speechSynthesis.speak(utterance);
  });

  // ---- Sample gallery: click a thumbnail to analyse it straight away
  document.querySelectorAll(".gallery-item").forEach((item) =>
    item.addEventListener("click", async () => {
      try {
        const response = await fetch(item.querySelector("img").src);
        const blob = await response.blob();
        choose(new File([blob], item.dataset.sample, { type: blob.type || "image/jpeg" }));
        form.requestSubmit();
      } catch (err) {
        showError(t.error_network);
      }
    })
  );

  function render(data) {
    placeholder.hidden = true;
    resultBox.hidden = false;
    lastResult = data;
    if ("speechSynthesis" in window) speechSynthesis.cancel();
    setListening(false);
    renderInsights(data);

    document.getElementById("cache-badge").hidden = !data.from_cache;
    document.getElementById("time-badge").textContent = data.elapsed_ms.toFixed(1) + " ms";

    const verdict = document.getElementById("verdict");
    const notSure = document.getElementById("not-sure");
    if (data.status === "ok") {
      verdict.hidden = false;
      notSure.hidden = true;
      verdict.classList.toggle("healthy", data.prediction.healthy);
      document.getElementById("verdict-crop").textContent = cropName(data.prediction);
      document.getElementById("verdict-condition").textContent =
        conditionName(data.prediction) + " · " + percent(data.prediction.confidence);
    } else {
      verdict.hidden = true;
      notSure.hidden = false;
      document.getElementById("not-sure-msg").textContent = t.not_sure_message;
    }

    renderAdvisory(data.status === "ok" ? data.advisory : null);
    document.getElementById("top-list").replaceChildren(...data.top.map(topItem));
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!selectedFile) return;
    button.disabled = true;
    button.textContent = t.btn_analysing;
    errorBox.hidden = true;
    try {
      const body = new FormData();
      body.append("image", selectedFile);
      body.append("lang", lang);
      const response = await fetch("/predict", { method: "POST", body });
      const data = await response.json();
      if (!response.ok) {
        showError(data.error || t.error_generic);
      } else {
        render(data);
      }
    } catch (err) {
      showError(t.error_network);
    } finally {
      button.disabled = false;
      button.textContent = t.btn_analyse;
    }
  });
})();
