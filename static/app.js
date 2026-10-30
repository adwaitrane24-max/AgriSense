// Upload page: preview the chosen image, POST it to /predict, render the result.
(function () {
  const form = document.getElementById("upload-form");
  const input = document.getElementById("file-input");
  const dropzone = document.getElementById("dropzone");
  const preview = document.getElementById("preview");
  const dropText = document.getElementById("drop-text");
  const button = document.getElementById("submit-btn");
  const errorBox = document.getElementById("error");
  const placeholder = document.getElementById("placeholder");
  const resultBox = document.getElementById("result");

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

  function render(data) {
    placeholder.hidden = true;
    resultBox.hidden = false;

    document.getElementById("cache-badge").hidden = !data.from_cache;
    document.getElementById("time-badge").textContent = data.elapsed_ms.toFixed(1) + " ms";

    const verdict = document.getElementById("verdict");
    const notSure = document.getElementById("not-sure");
    if (data.status === "ok") {
      verdict.hidden = false;
      notSure.hidden = true;
      verdict.classList.toggle("healthy", data.prediction.healthy);
      document.getElementById("verdict-crop").textContent = data.prediction.crop;
      document.getElementById("verdict-condition").textContent =
        data.prediction.condition + " · " + percent(data.prediction.confidence);
    } else {
      verdict.hidden = true;
      notSure.hidden = false;
      document.getElementById("not-sure-msg").textContent = data.message;
    }

    const list = document.getElementById("top-list");
    list.replaceChildren(
      ...data.top.map((item) => {
        const li = document.createElement("li");
        li.innerHTML =
          '<div class="top-row"><span></span><strong></strong></div>' +
          '<div class="bar"><div class="bar-fill"></div></div>';
        li.querySelector("span").textContent = item.crop + " — " + item.condition;
        li.querySelector("strong").textContent = percent(item.confidence);
        const fill = li.querySelector(".bar-fill");
        fill.style.width = percent(item.confidence);
        fill.classList.add(item.healthy ? "bar-hit" : "bar-miss");
        return li;
      })
    );
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!selectedFile) return;
    button.disabled = true;
    button.textContent = "Analysing…";
    errorBox.hidden = true;
    try {
      const body = new FormData();
      body.append("image", selectedFile);
      const response = await fetch("/predict", { method: "POST", body });
      const data = await response.json();
      if (!response.ok) {
        showError(data.error || "Something went wrong.");
      } else {
        render(data);
      }
    } catch (err) {
      showError("Could not reach the server.");
    } finally {
      button.disabled = false;
      button.textContent = "Analyse leaf";
    }
  });
})();
