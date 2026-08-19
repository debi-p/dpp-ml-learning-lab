# Neural Insight Studio Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the existing common visualizer into `Neural Insight Studio`, using the first deployed mockup style and real model traces from the API.

**Architecture:** Keep `common_model_api` as the trace provider and `model_flow_visualizer` as a static HTML/CSS/JS app. The visualizer renders model dropdown, mode selection, cinematic flow, playback controls, and inspection panels from the current trace data. Phase 1 remains the only fully supported model, but the UI should stay registry-driven.

**Tech Stack:** Plain HTML, CSS, JavaScript, FastAPI, existing NumPy model SDK.

---

## File Structure

- Modify: `/Users/debi.pradhan/Documents/ML/model_flow_visualizer/index.html`
  - Rename the product surface.
  - Move model selection to the header.
  - Replace old control panel layout with Training Mode / Ask Mode controls.
  - Keep required IDs used by `app.js`.
- Modify: `/Users/debi.pradhan/Documents/ML/model_flow_visualizer/style.css`
  - Apply the dark professional visual design from the accepted mockup.
  - Style logo, model dropdown, mode buttons, icon controls, speed slider, cinematic canvas, and inspection cards.
- Modify: `/Users/debi.pradhan/Documents/ML/model_flow_visualizer/app.js`
  - Add UI mode state: `training` and `ask`.
  - Make Training Mode call `/inspect-training-step`.
  - Make Ask Mode call `/inspect-forward`.
  - Update speed label to `Slow`, `Normal`, or `Fast`.
  - Keep canvas rendering from trace metadata when available.
- Modify: `/Users/debi.pradhan/Documents/ML/common_model_api/app.py`
  - Only if needed: ensure model metadata returned from `/models` includes display labels and supported modes.
- Modify: `/Users/debi.pradhan/Documents/ML/common_model_api/tests/test_app.py`
  - Add/adjust tests for model metadata if API changes.
- Create: `/Users/debi.pradhan/Documents/ML/model_flow_visualizer/visualizer_smoke_test.js`
  - Lightweight browser-free checks for expected UI IDs and labels.

---

## Task 1: Preserve Required DOM IDs While Updating Layout

**Files:**
- Modify: `/Users/debi.pradhan/Documents/ML/model_flow_visualizer/index.html`
- Test: `/Users/debi.pradhan/Documents/ML/model_flow_visualizer/visualizer_smoke_test.js`

- [ ] **Step 1: Add a smoke test that checks required IDs exist**

Create `/Users/debi.pradhan/Documents/ML/model_flow_visualizer/visualizer_smoke_test.js`:

```javascript
const fs = require("fs");
const html = fs.readFileSync("index.html", "utf8");

const requiredIds = [
  "apiBase",
  "apiStatus",
  "modelSelect",
  "messageInput",
  "correctLabel",
  "learningRate",
  "inspectForwardBtn",
  "inspectTrainingBtn",
  "previousBtn",
  "playBtn",
  "pauseBtn",
  "nextBtn",
  "replayBtn",
  "speedControl",
  "speedLabel",
  "predictionValue",
  "vectorShape",
  "hiddenShape",
  "lossValue",
  "stageTitle",
  "stageDescription",
  "flowLane",
  "flowSvg",
  "connectionLayer",
  "particleLayer",
  "boxInput",
  "boxAlgorithm",
  "boxOutput"
];

for (const id of requiredIds) {
  if (!html.includes(`id="${id}"`)) {
    throw new Error(`Missing required id: ${id}`);
  }
}

const requiredText = ["Neural Insight Studio", "Training Mode", "Ask Mode"];
for (const text of requiredText) {
  if (!html.includes(text)) {
    throw new Error(`Missing required text: ${text}`);
  }
}

console.log("visualizer smoke test passed");
```

- [ ] **Step 2: Run smoke test and verify it fails before layout update**

Run:

```bash
cd /Users/debi.pradhan/Documents/ML/model_flow_visualizer
node visualizer_smoke_test.js
```

Expected:

```text
Error: Missing required text: Neural Insight Studio
```

- [ ] **Step 3: Update `index.html` header and controls**

Edit `/Users/debi.pradhan/Documents/ML/model_flow_visualizer/index.html` so the top surface contains:

```html
<header class="studio-header">
  <div class="brand-lockup">
    <div class="studio-logo" aria-hidden="true">⌘</div>
    <div>
      <h1>Neural Insight Studio</h1>
      <p>Transparent model training and inference flow for learning how neural networks work internally.</p>
    </div>
  </div>
  <div class="api-status" id="apiStatus">API not checked</div>
</header>

<section class="studio-controls">
  <input id="apiBase" value="http://127.0.0.1:8010" aria-label="API Base URL" />
  <select id="modelSelect" aria-label="Select model"></select>
  <div class="mode-toggle" aria-label="Visualizer mode">
    <button id="inspectTrainingBtn" class="mode-button active" type="button">Training Mode</button>
    <button id="inspectForwardBtn" class="mode-button" type="button">Ask Mode</button>
  </div>
  <textarea id="messageInput" rows="3">Can we review the project deadline tomorrow?</textarea>
  <select id="correctLabel" aria-label="Expected category">
    <option>work</option>
    <option>personal</option>
    <option>promotion</option>
    <option>spam</option>
  </select>
  <input id="learningRate" type="number" value="0.1" min="0.001" max="1" step="0.001" aria-label="Learning rate" />
</section>
```

Keep the existing visualization IDs in the page so `app.js` does not break.

- [ ] **Step 4: Run smoke test and verify it passes**

Run:

```bash
cd /Users/debi.pradhan/Documents/ML/model_flow_visualizer
node visualizer_smoke_test.js
```

Expected:

```text
visualizer smoke test passed
```

---

## Task 2: Apply Accepted Visual Style

**Files:**
- Modify: `/Users/debi.pradhan/Documents/ML/model_flow_visualizer/style.css`

- [ ] **Step 1: Run CSS/JS syntax checks before styling**

Run:

```bash
cd /Users/debi.pradhan/Documents/ML/model_flow_visualizer
node --check app.js
node visualizer_smoke_test.js
```

Expected:

```text
visualizer smoke test passed
```

- [ ] **Step 2: Replace light dashboard variables with dark studio theme**

Update `:root` in `/Users/debi.pradhan/Documents/ML/model_flow_visualizer/style.css`:

```css
:root {
  color-scheme: dark;
  --bg: #050812;
  --panel: rgba(13, 23, 42, 0.88);
  --panel-strong: #0b1220;
  --ink: #eef7ff;
  --muted: #9db2c7;
  --line: rgba(125, 211, 252, 0.22);
  --blue: #38bdf8;
  --green: #34d399;
  --amber: #fbbf24;
  --red: #fb7185;
  --violet: #a78bfa;
  --shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
}
```

- [ ] **Step 3: Style the studio container and controls**

Add CSS for:

```css
body {
  background:
    radial-gradient(circle at 14% 16%, rgba(56, 189, 248, 0.18), transparent 28%),
    radial-gradient(circle at 78% 24%, rgba(167, 139, 250, 0.16), transparent 26%),
    linear-gradient(135deg, #050812, #07111f 52%, #0f1022);
  color: var(--ink);
}

.studio-header,
.studio-controls,
.content {
  max-width: 1180px;
  margin: 0 auto;
}

.studio-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 28px 30px 18px;
}

.brand-lockup {
  display: grid;
  grid-template-columns: 54px 1fr;
  gap: 14px;
  align-items: center;
}

.studio-logo {
  width: 54px;
  height: 54px;
  display: grid;
  place-items: center;
  border: 1px solid rgba(56, 189, 248, 0.42);
  border-radius: 14px;
  background: linear-gradient(135deg, rgba(56, 189, 248, 0.22), rgba(167, 139, 250, 0.24));
  box-shadow: 0 0 30px rgba(56, 189, 248, 0.18);
  color: #e0f2fe;
  font-size: 26px;
}

.studio-controls {
  display: grid;
  grid-template-columns: 220px minmax(260px, 1fr) auto minmax(260px, 1.2fr) 170px 120px;
  gap: 12px;
  align-items: center;
  padding: 0 30px 24px;
}
```

- [ ] **Step 4: Style icon controls**

Use text symbols initially, then replace with lucide icons later if the app adds an icon package:

```html
<button id="previousBtn" type="button" aria-label="Previous">‹</button>
<button id="playBtn" type="button" aria-label="Play">▶</button>
<button id="pauseBtn" type="button" aria-label="Pause">Ⅱ</button>
<button id="nextBtn" type="button" aria-label="Next">›</button>
<button id="replayBtn" type="button" aria-label="Replay">↻</button>
```

- [ ] **Step 5: Verify static checks**

Run:

```bash
cd /Users/debi.pradhan/Documents/ML/model_flow_visualizer
node --check app.js
node visualizer_smoke_test.js
```

Expected:

```text
visualizer smoke test passed
```

---

## Task 3: Add Mode State and Correct Button Behavior

**Files:**
- Modify: `/Users/debi.pradhan/Documents/ML/model_flow_visualizer/app.js`

- [ ] **Step 1: Add mode state**

Near the existing state variables, add:

```javascript
let activeMode = "training";
```

- [ ] **Step 2: Replace direct inspect button listeners**

Replace:

```javascript
document.querySelector("#inspectForwardBtn").addEventListener("click", () => inspect("forward"));
document.querySelector("#inspectTrainingBtn").addEventListener("click", () => inspect("training"));
```

with:

```javascript
document.querySelector("#inspectForwardBtn").addEventListener("click", () => setModeAndInspect("ask"));
document.querySelector("#inspectTrainingBtn").addEventListener("click", () => setModeAndInspect("training"));
```

- [ ] **Step 3: Add `setModeAndInspect`**

Add:

```javascript
function setModeAndInspect(mode) {
  activeMode = mode;
  document.querySelector("#inspectTrainingBtn").classList.toggle("active", mode === "training");
  document.querySelector("#inspectForwardBtn").classList.toggle("active", mode === "ask");
  correctLabel.disabled = mode === "ask";
  learningRate.disabled = mode === "ask";
  inspect(mode === "training" ? "training" : "forward");
}
```

- [ ] **Step 4: Verify syntax**

Run:

```bash
cd /Users/debi.pradhan/Documents/ML/model_flow_visualizer
node --check app.js
```

Expected: no output and exit code `0`.

---

## Task 4: Make Speed Control Human-Friendly

**Files:**
- Modify: `/Users/debi.pradhan/Documents/ML/model_flow_visualizer/index.html`
- Modify: `/Users/debi.pradhan/Documents/ML/model_flow_visualizer/app.js`

- [ ] **Step 1: Set speed slider to simple levels**

In `/Users/debi.pradhan/Documents/ML/model_flow_visualizer/index.html`, set:

```html
<input id="speedControl" type="range" min="1" max="5" value="2" step="1" />
<p class="hint" id="speedLabel">Slow</p>
```

- [ ] **Step 2: Add speed mapping**

In `/Users/debi.pradhan/Documents/ML/model_flow_visualizer/app.js`, add:

```javascript
const speedOptions = {
  1: { label: "Very Slow", delay: 2200 },
  2: { label: "Slow", delay: 1500 },
  3: { label: "Normal", delay: 950 },
  4: { label: "Fast", delay: 550 },
  5: { label: "Very Fast", delay: 250 },
};
```

- [ ] **Step 3: Update speed listener**

Replace:

```javascript
speedControl.addEventListener("input", () => {
  speedLabel.textContent = `Step delay: ${speedControl.value} ms`;
});
```

with:

```javascript
speedControl.addEventListener("input", () => {
  const option = speedOptions[speedControl.value] || speedOptions[3];
  speedLabel.textContent = option.label;
});
```

- [ ] **Step 4: Update playback delay function**

Find `getStageDelay`. Make sure it returns the mapped delay:

```javascript
function getStageDelay(stage) {
  const option = speedOptions[speedControl.value] || speedOptions[3];
  return stage?.delay || option.delay;
}
```

- [ ] **Step 5: Verify syntax**

Run:

```bash
cd /Users/debi.pradhan/Documents/ML/model_flow_visualizer
node --check app.js
```

Expected: no output and exit code `0`.

---

## Task 5: Update Teaching Copy Without Changing Model Logic

**Files:**
- Modify: `/Users/debi.pradhan/Documents/ML/model_flow_visualizer/app.js`
- Modify: `/Users/debi.pradhan/Documents/ML/model_flow_visualizer/index.html`

- [ ] **Step 1: Use beginner-friendly category names in visible UI**

Add this mapping to `app.js`:

```javascript
const labelDisplayNames = {
  work: "Work Email",
  personal: "Personal Email",
  promotion: "Promotional Email",
  spam: "Spam Email",
};

function displayLabel(label) {
  return labelDisplayNames[label] || label;
}
```

- [ ] **Step 2: Update correct label dropdown display**

In `updateCorrectLabelControl`, change:

```javascript
option.textContent = label;
```

to:

```javascript
option.textContent = displayLabel(label);
```

- [ ] **Step 3: Update static teaching copy**

Ensure the Dataset card says:

```text
Input: built-in email dataset
Source: spam and ham emails
Output: expected category
Categories: Spam Email, Personal Email, Work Email, Promotional Email
```

- [ ] **Step 4: Verify syntax and smoke test**

Run:

```bash
cd /Users/debi.pradhan/Documents/ML/model_flow_visualizer
node --check app.js
node visualizer_smoke_test.js
```

Expected:

```text
visualizer smoke test passed
```

---

## Task 6: Verify API and End-to-End Local Behavior

**Files:**
- Test only unless failures reveal necessary changes.

- [ ] **Step 1: Run Phase 1 tests**

Run:

```bash
cd /Users/debi.pradhan/Documents/ML/01_foundational_neural_network
python3 run_all_tests.py
```

Expected: all tests pass.

- [ ] **Step 2: Run API tests**

Run:

```bash
cd /Users/debi.pradhan/Documents/ML/common_model_api
python3 run_api_tests.py
```

Expected: all tests pass.

- [ ] **Step 3: Start API**

Run:

```bash
cd /Users/debi.pradhan/Documents/ML/common_model_api
uvicorn app:app --host 127.0.0.1 --port 8010
```

Expected:

```text
Uvicorn running on http://127.0.0.1:8010
```

- [ ] **Step 4: Start UI**

Run in another terminal:

```bash
cd /Users/debi.pradhan/Documents/ML/model_flow_visualizer
python3 serve_ui.py
```

Expected:

```text
Serving on http://127.0.0.1:8020
```

- [ ] **Step 5: Manual browser checks**

Open:

```text
http://127.0.0.1:8020/
```

Check:

- Model dropdown loads `dpp-email-classifier-small-v1`.
- Training Mode runs a training trace.
- Ask Mode runs a forward trace.
- Previous, Play, Pause, Next, and Replay work.
- Speed slider changes playback pace.
- Paused panel updates with input, algorithm, and output.
- No flowing text appears between boxes.

---

## Self-Review

- Spec coverage: The plan covers layout, model dropdown, modes, speed, icon controls, dataset wording, playback, and no flowing text.
- Placeholder scan: No `TBD`, `TODO`, or undefined future-only steps are present.
- Type consistency: Existing IDs from `app.js` are preserved; new mode values are `training` and `ask`.
