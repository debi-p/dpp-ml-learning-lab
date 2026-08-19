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
  "boxOutput",
  "datasetCard",
  "modeTitle",
  "modeCard",
  "gitaTracePanel",
  "gitaTraceTitle",
  "gitaSourceList",
  "gitaAnswerText",
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
