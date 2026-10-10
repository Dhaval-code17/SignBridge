const state = {
  stream: null,
  captureTimer: null,
  frameBuffer: [],
  modelBusy: false,
  cameraOn: false,
  canvas: document.createElement("canvas"),
  lastPrediction: null,
};

const $ = (id) => document.getElementById(id);

state.canvas.width = 224;
state.canvas.height = 224;


// ============================================================
// LOGGING
// ============================================================

function log(message) {
  const activityLog = $("activityLog");

  if (!activityLog) {
    return;
  }

  const row = document.createElement("div");

  row.className = "log-line";

  const now = new Date();

  const time = now.toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  row.innerHTML =
    `<span class="log-time">${time}</span>` +
    `<span class="log-text"></span>`;

  row.querySelector(".log-text").textContent =
    message;

  activityLog.prepend(row);
}


// ============================================================
// SYSTEM STATUS
// ============================================================

function setStatus(
  text,
  type = "ready"
) {
  if ($("systemStatus")) {
    $("systemStatus").textContent =
      text;
  }

  if ($("systemDot")) {
    $("systemDot").className =
      "status-dot" +
      (
        type === "live"
          ? " live"
          : type === "error"
            ? " error"
            : ""
      );
  }
}


// ============================================================
// MAIN OUTPUT RESET
// ============================================================

function resetOutput() {

  if ($("signSequence")) {
    $("signSequence").textContent =
      "—";
  }

  if ($("translation")) {
    $("translation").textContent =
      "Start the camera or upload a video.";
  }

  if ($("latency")) {
    $("latency").textContent =
      "—";
  }

  if ($("fps")) {
    $("fps").textContent =
      "—";
  }

  if ($("featureShape")) {
    $("featureShape").textContent =
      "[T, 960]";
  }

  if ($("inferenceBadge")) {
    $("inferenceBadge").textContent =
      "REAL MODEL 1";
  }

  if ($("speakBtn")) {
    $("speakBtn").disabled =
      true;
  }

  if ($("copyBtn")) {
    $("copyBtn").disabled =
      true;
  }

  state.lastPrediction = null;

  setLiveRecognitionState(
    "MODEL 1 READY",
    "Waiting for sign…",
    "Camera analysis will appear here.",
    "ready"
  );
}


// ============================================================
// BACKEND HEALTH CHECK
// ============================================================

async function checkBackend() {

  try {

    const response =
      await fetch(
        "/api/health",
        {
          cache: "no-store",
        }
      );

    if (!response.ok) {

      throw new Error(
        `HTTP ${response.status}`
      );
    }

    const data =
      await response.json();

    if (!data.ok) {

      throw new Error(
        data.error ||
        "Backend unavailable"
      );
    }

    if ($("inferenceBadge")) {
      $("inferenceBadge").textContent =
        "REAL MODEL 1";
    }

    log(
      `Real Model 1 connected on ${data.device}.`
    );

    log(
      `Recognition library: ` +
      `${data.reference_videos} training videos / ` +
      `${data.reference_glosses} glosses.`
    );

    setStatus(
      "Model 1 ready",
      "live"
    );

    return true;

  } catch (error) {

    if ($("inferenceBadge")) {
      $("inferenceBadge").textContent =
        "BACKEND OFF";
    }

    log(
      `Backend error: ${error.message}`
    );

    setStatus(
      "Backend offline",
      "error"
    );

    setLiveRecognitionState(
      "BACKEND OFFLINE",
      "—",
      "Start signbridge_server.py first.",
      "uncertain"
    );

    return false;
  }
}


// ============================================================
// LIVE RECOGNITION OVERLAY
// ============================================================

function createLiveRecognitionUI() {

  if (
    document.getElementById(
      "liveRecognitionOverlay"
    )
  ) {
    return;
  }

  const videoWrap =
    document.querySelector(
      ".video-wrap"
    );

  if (!videoWrap) {
    return;
  }

  // ----------------------------------------------------------
  // Styles
  // ----------------------------------------------------------

  const style =
    document.createElement(
      "style"
    );

  style.id =
    "liveRecognitionStyles";

  style.textContent = `
    #liveRecognitionOverlay {
      position: absolute;
      left: 14px;
      right: 14px;
      bottom: 14px;
      z-index: 20;
      padding: 14px 16px;
      border-radius: 15px;
      background: rgba(5, 10, 19, 0.90);
      border: 1px solid rgba(112, 225, 198, 0.24);
      backdrop-filter: blur(12px);
      box-shadow: 0 12px 32px rgba(0,0,0,.30);
      transition:
        border-color .2s ease,
        transform .2s ease;
    }

    #liveRecognitionOverlay.active {
      border-color: rgba(112,225,198,.62);
      transform: translateY(-1px);
    }

    #liveRecognitionOverlay.uncertain {
      border-color: rgba(255,193,92,.60);
    }

    #liveRecognitionOverlay.ready {
      border-color: rgba(255,255,255,.10);
    }

    .live-rec-status {
      display: flex;
      align-items: center;
      gap: 8px;
      color: #70e1c6;
      font-size: 10px;
      font-weight: 900;
      letter-spacing: .12em;
      margin-bottom: 7px;
    }

    .live-rec-dot {
      width: 8px;
      height: 8px;
      flex: 0 0 auto;
      border-radius: 999px;
      background: #70e1c6;
      box-shadow: 0 0 12px rgba(112,225,198,.75);
      animation: signbridgePulse 1.25s infinite;
    }

    .live-rec-sign {
      color: #fff;
      font-size: clamp(22px, 4vw, 34px);
      font-weight: 900;
      letter-spacing: -.035em;
      line-height: 1.04;
      text-transform: uppercase;
      word-break: break-word;
    }

    .live-rec-meta {
      color: #9eabc0;
      font-size: 11px;
      margin-top: 7px;
      line-height: 1.4;
    }

    @keyframes signbridgePulse {
      0%, 100% {
        opacity: 1;
        transform: scale(1);
      }

      50% {
        opacity: .45;
        transform: scale(.82);
      }
    }

    .live-rec-mini {
      display: inline-flex;
      margin-top: 8px;
      margin-right: 5px;
      padding: 4px 7px;
      border-radius: 999px;
      font-size: 9px;
      font-weight: 800;
      color: #d9dfeb;
      background: rgba(255,255,255,.06);
      border: 1px solid rgba(255,255,255,.07);
    }
  `;

  document.head.appendChild(
    style
  );

  // ----------------------------------------------------------
  // Overlay
  // ----------------------------------------------------------

  const overlay =
    document.createElement(
      "div"
    );

  overlay.id =
    "liveRecognitionOverlay";

  overlay.className =
    "ready";

  overlay.innerHTML = `
    <div class="live-rec-status">
      <span
        class="live-rec-dot"
      ></span>

      <span
        id="liveRecStatusText"
      >
        MODEL 1 READY
      </span>
    </div>

    <div
      class="live-rec-sign"
      id="liveRecSign"
    >
      Waiting for sign…
    </div>

    <div
      class="live-rec-meta"
      id="liveRecMeta"
    >
      Camera analysis will appear here.
    </div>

    <div
      id="liveRecTop"
    ></div>
  `;

  videoWrap.appendChild(
    overlay
  );
}


// ============================================================
// UPDATE LIVE OVERLAY
// ============================================================

function setLiveRecognitionState(
  status,
  sign,
  meta,
  type = "active",
  top5 = []
) {

  const overlay =
    document.getElementById(
      "liveRecognitionOverlay"
    );

  if (!overlay) {
    return;
  }

  overlay.className =
    type;

  const statusText =
    document.getElementById(
      "liveRecStatusText"
    );

  const signText =
    document.getElementById(
      "liveRecSign"
    );

  const metaText =
    document.getElementById(
      "liveRecMeta"
    );

  const topContainer =
    document.getElementById(
      "liveRecTop"
    );

  if (statusText) {

    statusText.textContent =
      status;
  }

  if (signText) {

    signText.textContent =
      sign;
  }

  if (metaText) {

    metaText.textContent =
      meta;
  }

  if (
    topContainer &&
    Array.isArray(top5) &&
    top5.length > 1
  ) {

    const remaining =
      top5
        .slice(1, 4)
        .map(
          item =>
            `<span class="live-rec-mini">` +
            `${escapeHtml(item.gloss)}` +
            ` ${Number(item.similarity).toFixed(3)}` +
            `</span>`
        )
        .join("");

    topContainer.innerHTML =
      remaining;

  } else if (topContainer) {

    topContainer.innerHTML =
      "";
  }
}


// ============================================================
// CAMERA START
// ============================================================

async function startCamera() {

  try {

    // --------------------------------------------------------
    // Backend check
    // --------------------------------------------------------

    const backendReady =
      await checkBackend();

    if (!backendReady) {

      log(
        "Camera blocked because Model 1 backend is offline."
      );

      return;
    }

    // --------------------------------------------------------
    // Camera API check
    // --------------------------------------------------------

    if (
      !navigator.mediaDevices ||
      !navigator.mediaDevices.getUserMedia
    ) {

      throw new Error(
        "Camera API is unavailable in this browser."
      );
    }

    // --------------------------------------------------------
    // Request camera
    // --------------------------------------------------------

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

            frameRate: {
              ideal: 30,
              max: 30,
            },
          },

          audio: false,
        }
      );

    // --------------------------------------------------------
    // Display camera
    // --------------------------------------------------------

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

    if ($("recordBtn")) {
      $("recordBtn").disabled =
        false;
    }

    $("stopBtn").disabled =
      false;

    state.cameraOn =
      true;

    state.modelBusy =
      false;

    state.frameBuffer =
      [];

    createLiveRecognitionUI();

    setStatus(
      "Live Model 1",
      "live"
    );

    setLiveRecognitionState(
      "MODEL 1 ACTIVE",
      "Waiting for sign…",
      "Collecting 16 frames at approximately 8 FPS.",
      "active"
    );

    log(
      "Camera started."
    );

    log(
      "Real-time Model 1 recognition started."
    );

    log(
      "Rolling window: 16 frames, 4-frame step."
    );

    startCapture();

  } catch (error) {

    setStatus(
      "Camera error",
      "error"
    );

    log(
      `Camera error: ${error.message}`
    );

    setLiveRecognitionState(
      "CAMERA ERROR",
      "—",
      error.message,
      "uncertain"
    );
  }
}


// ============================================================
// CAMERA STOP
// ============================================================

function stopCapture() {

  if (state.captureTimer) {

    clearInterval(
      state.captureTimer
    );

    state.captureTimer =
      null;
  }

  state.frameBuffer =
    [];

  state.modelBusy =
    false;
}


function stopCamera() {

  stopCapture();

  if (state.stream) {

    state.stream
      .getTracks()
      .forEach(
        track =>
          track.stop()
      );
  }

  state.stream =
    null;

  if ($("video")) {

    $("video").srcObject =
      null;

    $("video").style.display =
      "none";
  }

  $("cameraPlaceholder")
    ?.classList
    .remove(
      "hidden"
    );

  $("recordingPill")
    ?.classList
    .add(
      "hidden"
    );

  if ($("cameraBadge")) {
    $("cameraBadge").textContent =
      "OFF";
  }

  if ($("startBtn")) {
    $("startBtn").disabled =
      false;
  }

  if ($("recordBtn")) {
    $("recordBtn").disabled =
      true;
  }

  if ($("stopBtn")) {
    $("stopBtn").disabled =
      true;
  }

  state.cameraOn =
    false;

  setStatus(
    "Ready"
  );

  setLiveRecognitionState(
    "MODEL 1 READY",
    "Camera stopped",
    "Start the camera to recognize signs.",
    "ready"
  );

  log(
    "Camera stopped."
  );
}


// ============================================================
// CAPTURE FRAME
// ============================================================

function captureFrame() {

  const video =
    $("video");

  if (
    !video ||
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


// ============================================================
// ROLLING CAMERA CAPTURE
// ============================================================

function startCapture() {

  stopCapture();

  // 125 ms = approximately 8 FPS
  state.captureTimer =
    setInterval(
      () => {

        if (
          !state.cameraOn
        ) {
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

        // Prevent unlimited buffering.
        if (
          state.frameBuffer.length > 36
        ) {

          state.frameBuffer =
            state.frameBuffer.slice(
              -36
            );
        }

        // ----------------------------------------------------
        // Prediction window: 24 frames (~3s at 8 FPS)
        // Step: keep 16 frames, add 8 new frames
        // ----------------------------------------------------

        if (
          state.frameBuffer.length >= 24 &&
          !state.modelBusy
        ) {

          const frames =
            state.frameBuffer.slice(
              -24
            );

          // Keep 16 frames so the next prediction
          // uses 8 new frames (~1s step).
          state.frameBuffer =
            state.frameBuffer.slice(
              -16
            );

          state.modelBusy =
            true;

          processCameraFrames(
            frames
          ).finally(
            () => {

              state.modelBusy =
                false;
            }
          );
        }

      },
      125
    );
}


// ============================================================
// REAL CAMERA MODEL 1 + RECOGNITION
// ============================================================

async function processCameraFrames(
  frames
) {

  const requestStart =
    performance.now();

  setLiveRecognitionState(
    "ANALYZING LIVE SIGN",
    "Processing…",
    `${frames.length} frames → Model 1 → recognition`,
    "active"
  );

  try {

    const response =
      await fetch(
        "/api/model1",
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json"
          },

          body: JSON.stringify({
            frames
          })
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
      requestStart;

    // --------------------------------------------------------
    // Feature information
    // --------------------------------------------------------

    if ($("featureShape")) {

      $("featureShape")
        .textContent =
        `[${data.num_frames}, ${data.feature_dim}]`;
    }

    if ($("latency")) {

      $("latency")
        .textContent =
        `${Math.round(clientLatency)} ms`;
    }

    // IMPORTANT:
    // This is inference throughput, not camera FPS.
    if ($("fps")) {

      $("fps")
        .textContent =
        (
          data.num_frames /
          (
            clientLatency /
            1000
          )
        ).toFixed(1);
    }

    // --------------------------------------------------------
    // Prediction
    // --------------------------------------------------------

    const gloss =
      data.predicted_gloss ||
      "UNKNOWN";

    const similarity =
      Number(
        data.similarity || 0
      );

    const MIN_CONFIDENCE = 0.25;

    if (similarity < MIN_CONFIDENCE) {
      setLiveRecognitionState(
        "MODEL 1 ACTIVE",
        "Waiting for sign…",
        `Low confidence (${similarity.toFixed(3)}) · ${data.num_frames} frames`,
        "ready",
        data.top5 || []
      );
      return;
    }

    state.lastPrediction =
      gloss;

    if ($("signSequence")) {

      $("signSequence")
        .textContent =
        gloss;
    }

    if ($("translation")) {

      $("translation")
        .textContent =
        data.translated_text || gloss;
    }

    if ($("inferenceBadge")) {

      $("inferenceBadge")
        .textContent =
        "LIVE MODEL 1";
    }

    if ($("speakBtn")) {

      $("speakBtn")
        .disabled =
        false;
    }

    if ($("copyBtn")) {

      $("copyBtn")
        .disabled =
        false;
    }

    // --------------------------------------------------------
    // PROMINENT LIVE DISPLAY
    // --------------------------------------------------------

    setLiveRecognitionState(
      "● SIGN DETECTED",
      gloss,
      `Similarity: ${similarity.toFixed(4)} · ` +
      `${data.num_frames} frames · ` +
      `${Math.round(clientLatency)} ms`,
      "active",
      data.top5 || []
    );

    // --------------------------------------------------------
    // Top-5 result panel
    // --------------------------------------------------------

    const uploadResult =
      document.getElementById(
        "uploadResult"
      );

    if (
      uploadResult &&
      Array.isArray(data.top5)
    ) {

      const rows =
        data.top5
          .map(
            (
              item,
              index
            ) =>
              `
              <div class="top5-row">

                <span>
                  ${index + 1}.
                  ${escapeHtml(
                item.gloss
              )}
                </span>

                <strong>
                  ${Number(
                item.similarity
              ).toFixed(4)}
                </strong>

              </div>
              `
          )
          .join("");

      uploadResult.innerHTML =
        `
        <div class="upload-prediction">

          <div class="upload-small">
            LIVE PREDICTED SIGN
          </div>

          <div class="upload-big">
            ${escapeHtml(gloss)}
          </div>

          <div class="upload-small">
            Similarity:
            ${similarity.toFixed(4)}
          </div>

        </div>

        <div style="margin-top:14px">

          <div class="label">
            Top 5 matches
          </div>

          <div class="top5">
            ${rows}
          </div>

        </div>
        `;

      uploadResult.hidden =
        false;
    }

    // --------------------------------------------------------
    // Log
    // --------------------------------------------------------

    log(
      `LIVE SIGN: ${gloss} | ` +
      `similarity=${similarity.toFixed(4)} | ` +
      `${Math.round(clientLatency)} ms`
    );

    setStatus(
      "Recognizing live sign",
      "live"
    );

  } catch (error) {

    setLiveRecognitionState(
      "RECOGNITION ERROR",
      "—",
      error.message,
      "uncertain"
    );

    log(
      `Live recognition error: ${error.message}`
    );

    setStatus(
      "Recognition error",
      "error"
    );
  }
}


// ============================================================
// UPLOAD UI
// ============================================================

function createUploadUI() {

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
      border: 1px dashed rgba(255,255,255,.20);
      border-radius: 16px;
      padding: 28px;
      text-align: center;
      cursor: pointer;
      background: rgba(255,255,255,.025);
      transition: .15s ease;
    }

    .upload-drop:hover {
      border-color: #70e1c6;
      background: rgba(112,225,198,.035);
    }

    .upload-drop strong {
      display: block;
      margin-bottom: 7px;
      font-size: 17px;
    }

    .upload-drop span {
      color: #9eabc0;
      font-size: 12px;
      line-height: 1.5;
    }

    .upload-file-name {
      margin-top: 10px;
      color: #9eabc0;
      font-size: 12px;
      word-break: break-word;
    }

    .upload-result {
      margin-top: 17px;
    }

    .upload-prediction {
      padding: 18px;
      border-radius: 15px;
      border: 1px solid rgba(112,225,198,.22);
      background: rgba(112,225,198,.04);
    }

    .upload-small {
      color: #9eabc0;
      font-size: 11px;
      margin-bottom: 7px;
    }

    .upload-big {
      color: #fff;
      font-size: 30px;
      font-weight: 900;
      line-height: 1.05;
      letter-spacing: -.03em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }

    .top5 {
      display: grid;
      gap: 6px;
      margin-top: 8px;
    }

    .top5-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 9px 11px;
      border-radius: 9px;
      background: rgba(255,255,255,.04);
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

        <p class="eyebrow">
          VIDEO RECOGNITION
        </p>

        <h2>
          Upload a Sign Video
        </h2>

      </div>

      <span class="badge neutral">
        REAL MODEL 1
      </span>

    </div>

    <div
      class="upload-drop"
      id="uploadDrop"
      tabindex="0"
      role="button"
      aria-label="Choose sign language video"
    >

      <strong>
        Choose a video
      </strong>

      <span>
        MP4, MOV, AVI or WebM
        · Maximum 100 MB
        · Maximum 15 seconds
      </span>

      <div
        class="upload-file-name"
        id="uploadFileName"
      >
        No video selected
      </div>

    </div>

    <input
      type="file"
      id="videoFileInput"
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
      id="uploadResult"
      class="upload-result"
      hidden
    ></div>

    <div class="upload-note">

      The current recognizer uses the real Model 1
      960-D embeddings and the 5,686-video training
      reference library to identify an isolated sign.

      Similarity is a cosine similarity score,
      not a probability.

    </div>
  `;

  document
    .querySelector(
      ".app-shell"
    )
    .appendChild(
      card
    );


  const drop =
    $("uploadDrop");

  const input =
    $("videoFileInput");


  // ----------------------------------------------------------
  // Click
  // ----------------------------------------------------------

  drop.addEventListener(
    "click",
    () =>
      input.click()
  );


  // ----------------------------------------------------------
  // Keyboard
  // ----------------------------------------------------------

  drop.addEventListener(
    "keydown",
    event => {

      if (
        event.key === "Enter" ||
        event.key === " "
      ) {

        event.preventDefault();

        input.click();
      }

    }
  );


  // ----------------------------------------------------------
  // File selection
  // ----------------------------------------------------------

  input.addEventListener(
    "change",
    () => {

      const file =
        input.files?.[0];

      if (!file) {
        return;
      }

      $("uploadFileName")
        .textContent =
        `${file.name} · ` +
        `${(
          file.size /
          (
            1024 * 1024
          )
        ).toFixed(2)} MB`;

      $("recognizeVideoBtn")
        .disabled =
        false;

      $("uploadResult")
        .hidden =
        true;

      log(
        `Video selected: ${file.name}`
      );

    }
  );


  $("recognizeVideoBtn")
    .addEventListener(
      "click",
      recognizeUploadedVideo
    );


  $("clearVideoBtn")
    .addEventListener(
      "click",
      clearUpload
    );
}


// ============================================================
// UPLOAD VIDEO RECOGNITION
// ============================================================

async function recognizeUploadedVideo() {

  const input =
    $("videoFileInput");

  const file =
    input.files?.[0];

  if (!file) {
    return;
  }

  const button =
    $("recognizeVideoBtn");

  const result =
    $("uploadResult");


  button.disabled =
    true;

  button.textContent =
    "Recognizing…";

  result.hidden =
    true;


  setStatus(
    "Recognizing video",
    "live"
  );


  setLiveRecognitionState(
    "ANALYZING VIDEO",
    "Processing…",
    "Running real Model 1 recognition.",
    "active"
  );


  log(
    `Recognizing ${file.name}...`
  );


  try {

    const response =
      await fetch(
        "/api/recognize_video",
        {
          method: "POST",

          headers: {
            "Content-Type":
              file.type ||
              "video/mp4"
          },

          body: file
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


    // --------------------------------------------------------
    // Main result
    // --------------------------------------------------------

    const gloss =
      data.predicted_gloss ||
      "UNKNOWN";

    const similarity =
      Number(
        data.similarity || 0
      );


    $("signSequence")
      .textContent =
      gloss;


    $("translation")
      .textContent =
      data.translated_text || gloss;


    $("featureShape")
      .textContent =
      `[${data.feature_shape[0]}, ` +
      `${data.feature_shape[1]}]`;


    $("latency")
      .textContent =
      `${data.latency_ms} ms`;


    $("fps")
      .textContent =
      (
        data.num_frames /
        (
          data.latency_ms /
          1000
        )
      ).toFixed(1);


    $("speakBtn")
      .disabled =
      false;


    $("copyBtn")
      .disabled =
      false;


    $("inferenceBadge")
      .textContent =
      "REAL MODEL 1";


    // --------------------------------------------------------
    // Result display
    // --------------------------------------------------------

    renderUploadResult(
      data
    );


    setLiveRecognitionState(
      "SIGN DETECTED",
      gloss,
      `Similarity: ${similarity.toFixed(4)} · ` +
      `${data.num_frames} sampled frames · ` +
      `${data.latency_ms} ms`,
      "active",
      data.top5 || []
    );


    setStatus(
      "Recognition complete",
      "live"
    );


    log(
      `Predicted sign: ${gloss}`
    );


    log(
      `Similarity: ${similarity.toFixed(4)}`
    );


  } catch (error) {

    result.hidden =
      false;

    result.innerHTML =
      `
      <div class="result-block">

        <div class="label">
          Recognition error
        </div>

        <div>
          ${escapeHtml(
        error.message
      )}
        </div>

      </div>
      `;


    setLiveRecognitionState(
      "RECOGNITION FAILED",
      "—",
      error.message,
      "uncertain"
    );


    setStatus(
      "Recognition failed",
      "error"
    );


    log(
      `Recognition error: ${error.message}`
    );


  } finally {

    button.disabled =
      false;

    button.textContent =
      "Recognize Sign";
  }
}


// ============================================================
// UPLOAD RESULT RENDERER
// ============================================================

function renderUploadResult(
  data
) {

  const result =
    $("uploadResult");

  const gloss =
    data.predicted_gloss ||
    "UNKNOWN";

  const similarity =
    Number(
      data.similarity || 0
    );


  const rows =
    Array.isArray(data.top5)
      ? data.top5
        .map(
          (
            item,
            index
          ) =>
            `
              <div class="top5-row">

                <span>
                  ${index + 1}.
                  ${escapeHtml(
              item.gloss
            )}
                </span>

                <strong>
                  ${Number(
              item.similarity
            ).toFixed(4)}
                </strong>

              </div>
              `
        )
        .join("")
      : "";


  result.innerHTML =
    `
    <div class="upload-prediction">

      <div class="upload-small">
        PREDICTED SIGN
      </div>

      <div class="upload-big">
        ${escapeHtml(
      gloss
    )}
      </div>

      <div class="upload-small">
        Similarity:
        ${similarity.toFixed(4)}
        · ${data.num_frames}
        sampled frames
        · ${data.latency_ms} ms
      </div>

    </div>

    <div style="margin-top:14px">

      <div class="label">
        Top 5 matches
      </div>

      <div class="top5">
        ${rows}
      </div>

    </div>
    `;


  result.hidden =
    false;
}


// ============================================================
// CLEAR UPLOAD
// ============================================================

function clearUpload() {

  $("videoFileInput")
    .value =
    "";

  $("uploadFileName")
    .textContent =
    "No video selected";

  $("uploadResult")
    .hidden =
    true;

  $("recognizeVideoBtn")
    .disabled =
    true;

  resetOutput();

  log(
    "Uploaded video cleared."
  );
}


// ============================================================
// MOCK PIPELINE TEST
// ============================================================

function runMockPipeline() {

  const start =
    performance.now();


  $("runTestBtn")
    .disabled =
    true;


  $("testResult")
    .className =
    "test-result";


  $("testResult")
    .textContent =
    "Running…";


  log(
    "Mock pipeline test started."
  );


  setTimeout(
    () => {

      $("signSequence")
        .textContent =
        "THANK YOU";

      $("translation")
        .textContent =
        "Waiting for Model 3…";

      log(
        "Mock Model 2 output: THANK YOU"
      );

    },
    350
  );


  setTimeout(
    () => {

      $("translation")
        .textContent =
        "Thank you.";


      $("latency")
        .textContent =
        `${Math.round(
          performance.now() -
          start
        )} ms`;


      $("fps")
        .textContent =
        "8.0";


      $("speakBtn")
        .disabled =
        false;


      $("copyBtn")
        .disabled =
        false;


      $("testResult")
        .className =
        "test-result pass";


      $("testResult")
        .textContent =
        "PASS — UI mock flow works";


      $("runTestBtn")
        .disabled =
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
// TEXT TO SPEECH
// ============================================================

function speak() {

  const text =
    $("translation")
      .textContent
      .trim();


  if (
    !text ||
    text ===
    "Start the camera or upload a video." ||
    text.includes(
      "Waiting for Model"
    )
  ) {

    return;
  }


  if (
    !(
      "speechSynthesis"
      in window
    )
  ) {

    log(
      "Speech synthesis is unavailable."
    );

    return;
  }


  speechSynthesis.cancel();


  const utterance =
    new SpeechSynthesisUtterance(
      text
    );


  speechSynthesis.speak(
    utterance
  );


  log(
    "Speech playback requested."
  );
}


// ============================================================
// COPY TRANSLATION
// ============================================================

async function copyTranslation() {

  const text =
    $("translation")
      .textContent
      .trim();


  if (
    !text ||
    text ===
    "Start the camera or upload a video." ||
    text.includes(
      "Waiting for Model"
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
// HTML ESCAPE
// ============================================================

function escapeHtml(
  value
) {

  return String(
    value
  )
    .replaceAll(
      "&",
      "&amp;"
    )
    .replaceAll(
      "<",
      "&lt;"
    )
    .replaceAll(
      ">",
      "&gt;"
    )
    .replaceAll(
      '"',
      "&quot;"
    )
    .replaceAll(
      "'",
      "&#039;"
    );
}


// ============================================================
// EVENT LISTENERS
// ============================================================

$("startBtn")
  .addEventListener(
    "click",
    startCamera
  );


// ============================================================
// EXPLICIT 3-SECOND SIGN CLIP RECORDING
// ============================================================

async function recordSignClip() {
  if (!state.cameraOn || state.modelBusy) return;

  stopCapture();
  state.modelBusy = true;
  if ($("recordBtn")) $("recordBtn").disabled = true;

  setLiveRecognitionState(
    "RECORDING SIGN",
    "Perform your sign now…",
    "Capturing 24 frames over 3 seconds.",
    "active"
  );
  log("Started 3-second sign clip recording...");

  const clipFrames = [];
  const totalFrames = 24;
  const intervalMs = 125;

  for (let i = 0; i < totalFrames; i++) {
    await new Promise((resolve) => setTimeout(resolve, intervalMs));
    const frame = captureFrame();
    if (frame) clipFrames.push(frame);

    setLiveRecognitionState(
      "RECORDING SIGN",
      `Frame ${clipFrames.length} / ${totalFrames}`,
      "Keep signing...",
      "active"
    );
  }

  log(`Captured ${clipFrames.length} frames. Processing model pipeline...`);
  setLiveRecognitionState(
    "ANALYZING SIGN",
    "Processing Model 1 + Model 2 + Model 3...",
    "Executing sentence prototype matching...",
    "active"
  );

  try {
    const response = await fetch("/api/model1", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ frames: clipFrames }),
    });

    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || `HTTP ${response.status}`);
    }

    const gloss = data.predicted_gloss || "UNKNOWN";
    const similarity = Number(data.similarity || 0);

    if ($("signSequence")) $("signSequence").textContent = gloss;
    if ($("translation")) $("translation").textContent = data.translated_text || gloss;
    if ($("speakBtn")) $("speakBtn").disabled = false;
    if ($("copyBtn")) $("copyBtn").disabled = false;

    setLiveRecognitionState(
      "● SIGN RECOGNIZED",
      gloss,
      `Similarity: ${similarity.toFixed(4)} · ${data.num_frames} frames`,
      "active",
      data.top5 || []
    );
    log(`CLIP RECORDING RESULT: ${gloss} (similarity=${similarity.toFixed(4)})`);
  } catch (err) {
    log(`Clip recording error: ${err.message}`);
    setLiveRecognitionState("RECOGNITION ERROR", "—", err.message, "uncertain");
  } finally {
    state.modelBusy = false;
    if (state.cameraOn) {
      if ($("recordBtn")) $("recordBtn").disabled = false;
      startCapture();
    }
  }
}


$("stopBtn")
  .addEventListener(
    "click",
    stopCamera
  );


if ($("recordBtn")) {
  $("recordBtn").addEventListener("click", recordSignClip);
}


$("resetBtn")
  .addEventListener(
    "click",
    () => {

      stopCamera();

      resetOutput();

      createLiveRecognitionUI();

      checkBackend();

      log(
        "UI reset."
      );
    }
  );


$("runTestBtn")
  .addEventListener(
    "click",
    runMockPipeline
  );


$("speakBtn")
  .addEventListener(
    "click",
    speak
  );


$("copyBtn")
  .addEventListener(
    "click",
    copyTranslation
  );


$("clearLogBtn")
  .addEventListener(
    "click",
    () => {

      $("activityLog")
        .replaceChildren();

    }
  );


// ============================================================
// STARTUP
// ============================================================

createLiveRecognitionUI();

createUploadUI();

resetOutput();

log(
  "SignBridge UI loaded."
);

checkBackend();