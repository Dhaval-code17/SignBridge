const state = {
  stream: null,
  cameraOn: false,
  captureTimer: null,
  frameBuffer: [],
  modelBusy: false,
  canvas: document.createElement("canvas"),
};

const $ = (id) => document.getElementById(id);

state.canvas.width = 224;
state.canvas.height = 224;


// ============================================================
// LOGGING
// ============================================================

function log(message) {
  const now = new Date();

  const time = now.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  const row = document.createElement("div");

  row.className = "log-line";

  row.innerHTML =
    `<span class="log-time">${time}</span>` +
    `<span class="log-text"></span>`;

  row.querySelector(".log-text").textContent =
    message;

  $("activityLog").prepend(row);
}


// ============================================================
// STATUS
// ============================================================

function setSystemStatus(
  text,
  kind = "ready"
) {
  $("systemStatus").textContent =
    text;

  $("systemDot").className =
    "status-dot" +
    (
      kind === "live"
        ? " live"
        : kind === "error"
          ? " error"
          : ""
    );
}


// ============================================================
// OUTPUT RESET
// ============================================================

function resetOutput() {

  $("signSequence").textContent = "—";

  $("translation").textContent =
    "Start the camera or upload a video.";

  $("latency").textContent = "—";

  $("fps").textContent = "—";

  $("featureShape").textContent =
    "[T, 960]";

  $("inferenceBadge").textContent =
    "REAL MODEL 1";

  $("speakBtn").disabled = true;

  $("copyBtn").disabled = true;
}


// ============================================================
// BACKEND HEALTH
// ============================================================

async function checkBackend() {

  try {

    const response = await fetch(
      "/api/health",
      {
        cache: "no-store",
      }
    );

    const data =
      await response.json();

    if (!data.ok) {
      throw new Error(
        data.error ||
        "Backend unavailable"
      );
    }

    $("inferenceBadge").textContent =
      "REAL MODEL 1";

    $("featureShape").textContent =
      "[T, 960]";

    log(
      `Model 1 connected on ${data.device}.`
    );

    log(
      `Recognition library: ` +
      `${data.reference_videos} training videos / ` +
      `${data.reference_glosses} glosses.`
    );

    setSystemStatus(
      "Model 1 ready",
      "live"
    );

    return true;

  } catch (error) {

    $("inferenceBadge").textContent =
      "BACKEND OFF";

    log(
      `Backend not reachable: ${error.message}`
    );

    setSystemStatus(
      "Backend offline",
      "error"
    );

    return false;
  }
}


// ============================================================
// CAMERA
// ============================================================

async function startCamera() {

  if (
    !navigator.mediaDevices?.getUserMedia
  ) {

    setSystemStatus(
      "Camera unavailable",
      "error"
    );

    log(
      "Camera API unavailable."
    );

    return;
  }

  try {

    const backendReady =
      await checkBackend();

    if (!backendReady) {

      log(
        "Start signbridge_server.py first."
      );

      return;
    }

    state.stream =
      await navigator.mediaDevices.getUserMedia(
        {
          video: {
            facingMode: "user",
            width: {
              ideal: 1280,
            },
            height: {
              ideal: 720,
            },
          },
          audio: false,
        }
      );

    $("video").srcObject =
      state.stream;

    $("video").style.display =
      "block";

    $("cameraPlaceholder")
      .classList.add(
        "hidden"
      );

    $("recordingPill")
      .classList.remove(
        "hidden"
      );

    $("cameraBadge").textContent =
      "ON";

    $("startBtn").disabled =
      true;

    $("stopBtn").disabled =
      false;

    state.cameraOn = true;

    state.frameBuffer = [];

    setSystemStatus(
      "Live Model 1",
      "live"
    );

    log(
      "Camera started."
    );

    log(
      "Capturing 16-frame windows at ~8 FPS."
    );

    startModel1Capture();

  } catch (error) {

    setSystemStatus(
      "Camera permission needed",
      "error"
    );

    log(
      `Camera error: ${error.message}`
    );
  }
}


function stopModel1Capture() {

  if (state.captureTimer) {

    clearInterval(
      state.captureTimer
    );

    state.captureTimer = null;
  }

  state.frameBuffer = [];
}


function stopCamera() {

  stopModel1Capture();

  if (state.stream) {

    state.stream
      .getTracks()
      .forEach(
        (track) =>
          track.stop()
      );
  }

  state.stream = null;

  $("video").srcObject =
    null;

  $("video").style.display =
    "none";

  $("cameraPlaceholder")
    .classList.remove(
      "hidden"
    );

  $("recordingPill")
    .classList.add(
      "hidden"
    );

  $("cameraBadge").textContent =
    "OFF";

  $("startBtn").disabled =
    false;

  $("stopBtn").disabled =
    true;

  state.cameraOn = false;

  state.modelBusy = false;

  setSystemStatus(
    "Ready"
  );

  log(
    "Camera stopped."
  );
}


function captureFrame() {

  const video =
    $("video");

  if (
    !state.cameraOn ||
    video.readyState < 2
  ) {

    return null;
  }

  const ctx =
    state.canvas.getContext(
      "2d"
    );

  ctx.drawImage(
    video,
    0,
    0,
    224,
    224
  );

  return state.canvas.toDataURL(
    "image/jpeg",
    0.78
  );
}


function startModel1Capture() {

  stopModel1Capture();

  state.captureTimer =
    setInterval(
      () => {

        if (state.modelBusy) {
          return;
        }

        const frame =
          captureFrame();

        if (!frame) {
          return;
        }

        state.frameBuffer.push(
          frame
        );

        if (
          state.frameBuffer.length >= 16
        ) {

          const frames =
            state.frameBuffer.splice(
              0,
              16
            );

          state.modelBusy = true;

          processModel1(
            frames
          ).finally(
            () => {
              state.modelBusy = false;
            }
          );
        }

      },
      125
    );
}


// ============================================================
// REAL MODEL 1 CAMERA INFERENCE
// ============================================================

async function processModel1(
  frames
) {

  const t0 =
    performance.now();

  try {

    const response =
      await fetch(
        "/api/model1",
        {
          method: "POST",
          headers: {
            "Content-Type":
              "application/json",
          },
          body: JSON.stringify({
            frames,
          }),
        }
      );

    const data =
      await response.json();

    if (
      !response.ok ||
      !data.ok
    ) {

      throw new Error(
        data.error ||
        `HTTP ${response.status}`
      );
    }

    const clientLatency =
      performance.now() -
      t0;

    $("featureShape").textContent =
      `[${data.num_frames}, ${data.feature_dim}]`;

    $("latency").textContent =
      `${Math.round(clientLatency)} ms`;

    $("fps").textContent =
      (
        data.num_frames /
        (clientLatency / 1000)
      ).toFixed(1);

    $("signSequence").textContent =
      "FEATURES READY";

    $("translation").textContent =
      "Waiting for Model 2…";

    $("speakBtn").disabled =
      true;

    $("copyBtn").disabled =
      true;

    log(
      `REAL Model 1: ` +
      `[${data.num_frames},${data.feature_dim}] ` +
      `float32 — ` +
      `${data.latency_ms} ms`
    );

  } catch (error) {

    log(
      `Model 1 error: ${error.message}`
    );

    setSystemStatus(
      "Model 1 error",
      "error"
    );
  }
}


// ============================================================
// UPLOAD UI
// ============================================================

function addUploadUI() {

  if (
    document.getElementById(
      "videoUploadCard"
    )
  ) {

    return;
  }

  const style =
    document.createElement(
      "style"
    );

  style.textContent = `
    .upload-card {
      margin-top: 18px;
      padding: 20px;
    }

    .upload-drop {
      border: 1px dashed rgba(255,255,255,.18);
      border-radius: 16px;
      padding: 26px;
      text-align: center;
      background: rgba(255,255,255,.025);
      cursor: pointer;
      transition: .15s ease;
    }

    .upload-drop:hover {
      border-color: rgba(112,225,198,.55);
      background: rgba(112,225,198,.035);
    }

    .upload-drop strong {
      display: block;
      font-size: 16px;
      margin-bottom: 6px;
    }

    .upload-drop span {
      color: #9eabc0;
      font-size: 12px;
    }

    .upload-file-name {
      margin-top: 10px;
      color: #9eabc0;
      font-size: 12px;
      word-break: break-word;
    }

    .upload-result {
      display: grid;
      gap: 10px;
      margin-top: 16px;
    }

    .upload-prediction {
      padding: 18px;
      border: 1px solid rgba(112,225,198,.2);
      border-radius: 15px;
      background: rgba(112,225,198,.04);
    }

    .upload-prediction .small {
      color: #9eabc0;
      font-size: 11px;
      margin-bottom: 7px;
    }

    .upload-prediction .big {
      font-size: 28px;
      font-weight: 850;
      letter-spacing: -.03em;
    }

    .top5 {
      display: grid;
      gap: 6px;
    }

    .top5-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 8px 10px;
      background: rgba(255,255,255,.035);
      border-radius: 9px;
      font-size: 12px;
    }

    .upload-note {
      color: #9eabc0;
      font-size: 11px;
      line-height: 1.5;
      margin-top: 12px;
    }
  `;

  document.head.appendChild(
    style
  );

  const card =
    document.createElement(
      "section"
    );

  card.id =
    "videoUploadCard";

  card.className =
    "card upload-card";

  card.innerHTML = `
    <div class="card-header">
      <div>
        <p class="eyebrow">VIDEO RECOGNITION</p>
        <h2>Upload a Sign Video</h2>
      </div>
      <span class="badge neutral">REAL MODEL 1</span>
    </div>

    <div
      class="upload-drop"
      id="uploadDrop"
      role="button"
      tabindex="0"
    >
      <strong>Choose a video</strong>
      <span>
        MP4, MOV, AVI or WebM · maximum 100 MB ·
        maximum 15 seconds
      </span>

      <div
        class="upload-file-name"
        id="uploadFileName"
      >
        No video selected
      </div>
    </div>

    <input
      id="videoFileInput"
      type="file"
      accept="video/*"
      hidden
    >

    <div class="controls">
      <button
        class="btn primary"
        id="recognizeVideoBtn"
        disabled
      >
        Recognize Sign
      </button>
      <button
        class="btn ghost"
        id="clearVideoBtn"
      >
        Clear
      </button>
    </div>

    <div
      class="upload-result"
      id="uploadResult"
      hidden
    ></div>

    <div class="upload-note">
      The current recognizer matches the uploaded isolated-sign
      video against the Model 1 training reference library.
      The similarity score is not a probability.
      This is Model 1 recognition only; Model 2 and Model 3
      are not connected yet.
    </div>
  `;

  document
    .querySelector(".app-shell")
    .appendChild(card);

  const drop =
    document.getElementById(
      "uploadDrop"
    );

  const input =
    document.getElementById(
      "videoFileInput"
    );

  drop.addEventListener(
    "click",
    () => input.click()
  );

  drop.addEventListener(
    "keydown",
    (event) => {

      if (
        event.key === "Enter" ||
        event.key === " "
      ) {

        event.preventDefault();

        input.click();
      }
    }
  );

  input.addEventListener(
    "change",
    () => {

      const file =
        input.files?.[0];

      if (!file) {
        return;
      }

      $("uploadFileName").textContent =
        `${file.name} · ` +
        `${(
          file.size /
          (1024 * 1024)
        ).toFixed(2)} MB`;

      $("recognizeVideoBtn").disabled =
        false;

      $("uploadResult").hidden =
        true;

      log(
        `Video selected: ${file.name}`
      );
    }
  );

  document
    .getElementById(
      "recognizeVideoBtn"
    )
    .addEventListener(
      "click",
      recognizeUploadedVideo
    );

  document
    .getElementById(
      "clearVideoBtn"
    )
    .addEventListener(
      "click",
      clearUploadedVideo
    );
}


// ============================================================
// UPLOAD RECOGNITION
// ============================================================

async function recognizeUploadedVideo() {

  const input =
    document.getElementById(
      "videoFileInput"
    );

  const file =
    input.files?.[0];

  if (!file) {
    return;
  }

  const button =
    document.getElementById(
      "recognizeVideoBtn"
    );

  const resultBox =
    document.getElementById(
      "uploadResult"
    );

  button.disabled = true;

  button.textContent =
    "Recognizing…";

  resultBox.hidden =
    true;

  setSystemStatus(
    "Recognizing video",
    "live"
  );

  log(
    `Uploading ${file.name}...`
  );

  const t0 =
    performance.now();

  try {

    const response =
      await fetch(
        "/api/recognize_video",
        {
          method: "POST",
          headers: {
            "Content-Type":
              file.type ||
              "video/mp4",
          },
          body: file,
        }
      );

    const data =
      await response.json();

    if (
      !response.ok ||
      !data.ok
    ) {

      throw new Error(
        data.error ||
        `HTTP ${response.status}`
      );
    }

    const elapsed =
      performance.now() -
      t0;

    $("signSequence").textContent =
      data.predicted_gloss;

    $("translation").textContent =
      data.predicted_gloss;

    $("latency").textContent =
      `${Math.round(elapsed)} ms`;

    $("fps").textContent =
      (
        data.num_frames /
        (
          data.latency_ms /
          1000
        )
      ).toFixed(1);

    $("featureShape").textContent =
      `[${data.feature_shape[0]}, ` +
      `${data.feature_shape[1]}]`;

    $("speakBtn").disabled =
      false;

    $("copyBtn").disabled =
      false;

    renderUploadResult(
      data
    );

    setSystemStatus(
      "Recognition complete",
      "live"
    );

    log(
      `Recognized sign: ` +
      `${data.predicted_gloss}`
    );

    log(
      `Similarity: ` +
      `${data.similarity}`
    );

  } catch (error) {

    resultBox.hidden =
      false;

    resultBox.innerHTML = `
      <div class="result-block">
        <div class="label">Recognition error</div>
        <div>${escapeHtml(error.message)}</div>
      </div>
    `;

    log(
      `Video recognition error: ` +
      `${error.message}`
    );

    setSystemStatus(
      "Recognition failed",
      "error"
    );

  } finally {

    button.disabled =
      false;

    button.textContent =
      "Recognize Sign";
  }
}


// ============================================================
// UPLOAD RESULT
// ============================================================

function renderUploadResult(
  data
) {

  const box =
    document.getElementById(
      "uploadResult"
    );

  const rows =
    data.top5
      .map(
        (item, index) =>
          `
          <div class="top5-row">
            <span>
              ${index + 1}.
              ${escapeHtml(item.gloss)}
            </span>
            <strong>
              ${item.similarity}
            </strong>
          </div>
          `
      )
      .join("");

  box.innerHTML = `
    <div class="upload-prediction">
      <div class="small">
        PREDICTED SIGN
      </div>

      <div class="big">
        ${escapeHtml(
    data.predicted_gloss
  )}
      </div>

      <div class="small">
        Similarity: ${data.similarity}
        · ${data.num_frames} sampled frames
        · ${data.latency_ms} ms server inference
      </div>
    </div>

    <div>
      <div class="label">
        Top 5 matches
      </div>

      <div class="top5">
        ${rows}
      </div>
    </div>
  `;

  box.hidden =
    false;
}


// ============================================================
// CLEAR UPLOAD
// ============================================================

function clearUploadedVideo() {

  const input =
    document.getElementById(
      "videoFileInput"
    );

  const name =
    document.getElementById(
      "uploadFileName"
    );

  const result =
    document.getElementById(
      "uploadResult"
    );

  input.value = "";

  name.textContent =
    "No video selected";

  result.hidden =
    true;

  document.getElementById(
    "recognizeVideoBtn"
  ).disabled = true;

  resetOutput();

  log(
    "Uploaded video cleared."
  );
}


// ============================================================
// ESCAPE HTML
// ============================================================

function escapeHtml(
  value
) {

  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}


// ============================================================
// MOCK PIPELINE
// ============================================================

function runMockPipeline() {

  const t0 =
    performance.now();

  $("runTestBtn").disabled =
    true;

  $("testResult").className =
    "test-result";

  $("testResult").textContent =
    "Running…";

  log(
    "Mock pipeline test started."
  );

  window.setTimeout(
    () => {

      $("signSequence").textContent =
        "THANK YOU";

      log(
        "Mock Model 2 output: THANK YOU"
      );

    },
    350
  );

  window.setTimeout(
    () => {

      $("translation").textContent =
        "Thank you.";

      $("latency").textContent =
        `${Math.round(
          performance.now() -
          t0
        )} ms`;

      $("fps").textContent =
        "8.0";

      $("speakBtn").disabled =
        false;

      $("copyBtn").disabled =
        false;

      $("testResult").className =
        "test-result pass";

      $("testResult").textContent =
        "PASS — UI mock flow works";

      $("runTestBtn").disabled =
        false;

      log(
        "Mock Model 3 output: Thank you."
      );

      log(
        "Mock pipeline test passed."
      );

    },
    850
  );
}


// ============================================================
// TTS
// ============================================================

function speak() {

  const text =
    $("translation")
      .textContent
      .trim();

  if (
    !text ||
    text === "Start the camera or upload a video." ||
    text.includes(
      "Waiting for Model 2"
    )
  ) {

    return;
  }

  if (
    !("speechSynthesis" in window)
  ) {

    log(
      "Speech synthesis unavailable."
    );

    return;
  }

  speechSynthesis.cancel();

  speechSynthesis.speak(
    new SpeechSynthesisUtterance(
      text
    )
  );

  log(
    "Speech playback requested."
  );
}


// ============================================================
// COPY
// ============================================================

async function copyTranslation() {

  const text =
    $("translation")
      .textContent
      .trim();

  if (
    !text ||
    text === "Start the camera or upload a video." ||
    text.includes(
      "Waiting for Model 2"
    )
  ) {

    return;
  }

  try {

    await navigator.clipboard.writeText(
      text
    );

    log(
      "Translation copied."
    );

  } catch (error) {

    log(
      `Copy failed: ${error.message}`
    );
  }
}


// ============================================================
// EVENT LISTENERS
// ============================================================

$("startBtn").addEventListener(
  "click",
  startCamera
);

$("stopBtn").addEventListener(
  "click",
  stopCamera
);

$("resetBtn").addEventListener(
  "click",
  () => {

    stopCamera();

    resetOutput();

    checkBackend();

    log(
      "UI reset."
    );
  }
);

$("runTestBtn").addEventListener(
  "click",
  runMockPipeline
);

$("speakBtn").addEventListener(
  "click",
  speak
);

$("copyBtn").addEventListener(
  "click",
  copyTranslation
);

$("clearLogBtn").addEventListener(
  "click",
  () =>
    $("activityLog")
      .replaceChildren()
);


// ============================================================
// STARTUP
// ============================================================

resetOutput();

addUploadUI();

log(
  "SignBridge UI loaded."
);

checkBackend();