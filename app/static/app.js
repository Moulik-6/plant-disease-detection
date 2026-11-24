/**
 * AgriScan Client Application
 * Mobile-first responsive frontend logic and localization engine.
 */

let currentLang = "en";
let stringsData = {};
let selectedFile = null;
let currentPrediction = null;

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", async () => {
  await loadStrings();
  initUIEvents();
  initLanguageSelector();
});

/**
 * Loads strings.json for multi-language support (English, Telugu, Hindi).
 */
async function loadStrings() {
  try {
    const res = await fetch("/strings.json");
    if (res.ok) {
      stringsData = await res.json();
    }
  } catch (e) {
    console.warn("Could not load external strings.json, using fallback dictionary.", e);
  }
  applyTranslations();
}

/**
 * Updates DOM text contents based on current language.
 */
function applyTranslations() {
  const langTable = stringsData[currentLang] || stringsData["en"] || {};
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (langTable[key]) {
      el.textContent = langTable[key];
    }
  });

  // Update dynamic elements if results are currently visible
  if (currentPrediction) {
    updateResultCard(currentPrediction);
  }
}

function getTranslation(key, fallback = "") {
  const langTable = stringsData[currentLang] || stringsData["en"] || {};
  return langTable[key] || fallback;
}

/**
 * Wires up UI event handlers.
 */
function initUIEvents() {
  const btnCamera = document.getElementById("btn-camera");
  const btnUpload = document.getElementById("btn-upload");
  const cameraInput = document.getElementById("camera-input");
  const fileInput = document.getElementById("file-input");
  const btnClassify = document.getElementById("btn-classify");
  const tabDiagnose = document.getElementById("tab-btn-diagnose");
  const tabHistory = document.getElementById("tab-btn-history");
  const btnClearHistory = document.getElementById("btn-clear-history");
  const gradcamToggle = document.getElementById("gradcam-toggle");

  // Camera / Upload triggers
  btnCamera.addEventListener("click", () => cameraInput.click());
  btnUpload.addEventListener("click", () => fileInput.click());

  cameraInput.addEventListener("change", (e) => handleFileSelect(e.target.files[0]));
  fileInput.addEventListener("change", (e) => handleFileSelect(e.target.files[0]));

  // Classify trigger
  btnClassify.addEventListener("click", handleClassification);

  // Tab switching
  tabDiagnose.addEventListener("click", () => switchTab("diagnose"));
  tabHistory.addEventListener("click", () => {
    switchTab("history");
    loadHistory();
  });

  // History clearing
  btnClearHistory.addEventListener("click", handleClearHistory);

  // Grad-CAM toggle
  gradcamToggle.addEventListener("change", (e) => {
    const viewers = document.getElementById("gradcam-viewers");
    viewers.style.display = e.target.checked ? "grid" : "none";
  });
}

/**
 * Initializes language dropdown.
 */
function initLanguageSelector() {
  const langSelect = document.getElementById("lang-select");
  langSelect.addEventListener("change", (e) => {
    currentLang = e.target.value;
    applyTranslations();
  });
}

/**
 * Switches active view between Diagnose and History.
 */
function switchTab(tabName) {
  const tabDiagnoseBtn = document.getElementById("tab-btn-diagnose");
  const tabHistoryBtn = document.getElementById("tab-btn-history");
  const panelDiagnose = document.getElementById("tab-diagnose");
  const panelHistory = document.getElementById("tab-history");

  if (tabName === "diagnose") {
    tabDiagnoseBtn.classList.add("active");
    tabDiagnoseBtn.setAttribute("aria-selected", "true");
    tabHistoryBtn.classList.remove("active");
    tabHistoryBtn.setAttribute("aria-selected", "false");
    panelDiagnose.classList.add("active");
    panelHistory.classList.remove("active");
  } else {
    tabHistoryBtn.classList.add("active");
    tabHistoryBtn.setAttribute("aria-selected", "true");
    tabDiagnoseBtn.classList.remove("active");
    tabDiagnoseBtn.setAttribute("aria-selected", "false");
    panelHistory.classList.add("active");
    panelDiagnose.classList.remove("active");
  }
}

/**
 * Handles image file selection and generates preview.
 */
function handleFileSelect(file) {
  if (!file) return;

  selectedFile = file;
  const previewWrapper = document.getElementById("preview-wrapper");
  const imagePreview = document.getElementById("image-preview");
  const resultsCard = document.getElementById("results-card");

  // Hide old results
  resultsCard.style.display = "none";
  currentPrediction = null;

  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target.result;
    previewWrapper.style.display = "block";
    previewWrapper.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };
  reader.readAsDataURL(file);
}

/**
 * Executes API diagnosis request.
 */
async function handleClassification() {
  if (!selectedFile) return;

  const btnClassify = document.getElementById("btn-classify");
  const btnText = document.getElementById("classify-btn-text");
  const originalText = getTranslation("classify_button", "Analyze Crop Health");
  const analyzingText = getTranslation("analyzing", "Analyzing leaf symptoms...");

  btnClassify.disabled = true;
  btnText.innerHTML = `<span class="spinner"></span> ${analyzingText}`;

  const formData = new FormData();
  formData.append("image", selectedFile);
  formData.append("gradcam", "true");

  try {
    const res = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || getTranslation("error_upload", "Failed to process image"));
    }

    const data = await res.json();
    currentPrediction = data;
    updateResultCard(data);

    const resultsCard = document.getElementById("results-card");
    resultsCard.style.display = "block";
    resultsCard.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    alert(err.message || getTranslation("error_upload", "An error occurred"));
  } finally {
    btnClassify.disabled = false;
    btnText.textContent = originalText;
  }
}

/**
 * Populates results card with prediction metrics and Grad-CAM preview.
 */
function updateResultCard(pred) {
  const badge = document.getElementById("result-status-badge");
  const latency = document.getElementById("result-latency");
  const primaryTitle = document.getElementById("result-primary-title");
  const cropMeta = document.getElementById("result-crop-meta");
  const confVal = document.getElementById("result-confidence-val");
  const progFill = document.getElementById("result-progress-fill");
  const diffContainer = document.getElementById("diff-list-container");
  const gradcamToggle = document.getElementById("gradcam-toggle");
  const gradcamOrig = document.getElementById("gradcam-orig-img");
  const gradcamOverlay = document.getElementById("gradcam-overlay-img");
  const gradcamViewers = document.getElementById("gradcam-viewers");

  // Status Badge
  badge.className = "status-badge";
  let statusText = "";
  if (pred.status === "confident") {
    badge.classList.add("status-confident");
    statusText = getTranslation("status_confident", "Confident Diagnosis");
  } else if (pred.status === "uncertain") {
    badge.classList.add("status-uncertain");
    statusText = getTranslation("status_uncertain", "Uncertain - Verify Symptoms");
  } else {
    badge.classList.add("status-not_recognised");
    statusText = getTranslation("status_not_recognised", "Condition Not Recognised");
  }
  badge.textContent = statusText;

  // Latency & Title
  const latMs = pred.latency_ms || pred.inference_latency_ms || 0;
  latency.textContent = `${latMs} ms`;
  primaryTitle.textContent = `${pred.crop} — ${pred.condition}`;

  const healthText = pred.healthy
    ? `🌿 ${getTranslation("health_status_healthy", "Healthy Tissue")}`
    : `⚠️ ${getTranslation("health_status_diseased", "Pathology Detected")}`;
  cropMeta.textContent = `${getTranslation("crop_label", "Crop")}: ${pred.crop} | ${healthText}`;

  // Confidence bar
  const confPct = Math.round(pred.confidence * 100);
  confVal.textContent = `${confPct}%`;
  progFill.style.width = `${Math.min(100, Math.max(0, confPct))}%`;
  progFill.className = `progress-fill ${pred.status}`;

  // Differential Diagnoses (Top 3)
  diffContainer.innerHTML = "";
  const topList = pred.top3 || pred.top_3 || [];
  topList.forEach((item, index) => {
    const row = document.createElement("div");
    row.className = "diff-item";
    const itemPct = Math.round((item.confidence || item.probability || 0) * 100);
    row.innerHTML = `
      <div class="diff-name">
        <span>${index + 1}. ${item.crop} — ${item.condition}</span>
        ${item.healthy ? " (Healthy)" : ""}
      </div>
      <div class="diff-prob">${itemPct}%</div>
    `;
    diffContainer.appendChild(row);
  });

  // Grad-CAM visualization
  const imagePreview = document.getElementById("image-preview");
  gradcamOrig.src = imagePreview.src;

  if (pred.gradcam_image) {
    gradcamOverlay.src = pred.gradcam_image;
    document.querySelector(".gradcam-section").style.display = "block";
  } else {
    document.querySelector(".gradcam-section").style.display = "none";
  }

  // Keep toggle state in sync
  gradcamViewers.style.display = gradcamToggle.checked ? "grid" : "none";
}

/**
 * Loads recent history scans from /history endpoint.
 */
async function loadHistory() {
  const container = document.getElementById("history-container");
  try {
    const res = await fetch("/history?limit=50");
    if (!res.ok) throw new Error("Could not load history");

    const data = await res.json();
    const items = Array.isArray(data) ? data : data.items || [];

    if (items.length === 0) {
      container.innerHTML = `<div class="empty-state">${getTranslation("history_empty", "No prior scans recorded yet.")}</div>`;
      return;
    }

    container.innerHTML = "";
    items.forEach((item) => {
      const card = document.createElement("div");
      card.className = "history-item";

      let badgeClass = "status-confident";
      if (item.status === "uncertain") badgeClass = "status-uncertain";
      if (item.status === "not_recognised") badgeClass = "status-not_recognised";

      const confPct = Math.round(item.confidence * 100);
      const dt = item.timestamp ? new Date(item.timestamp).toLocaleString() : "";

      card.innerHTML = `
        <div class="history-item-top">
          <strong>${item.crop || item.predicted_class} — ${item.condition || ""}</strong>
          <span class="status-badge ${badgeClass}" style="font-size: 0.7rem; padding: 2px 6px;">
            ${item.status} (${confPct}%)
          </span>
        </div>
        <div class="history-item-date">${dt} • Model: ${item.model_name}</div>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    container.innerHTML = `<div class="empty-state text-danger">${err.message}</div>`;
  }
}

/**
 * Deletes all history from SQLite via DELETE /history.
 */
async function handleClearHistory() {
  if (!confirm("Are you sure you want to clear all diagnostic history?")) return;

  try {
    const res = await fetch("/history", { method: "DELETE" });
    if (res.ok) {
      await loadHistory();
    }
  } catch (err) {
    alert("Failed to clear history: " + err.message);
  }
}
