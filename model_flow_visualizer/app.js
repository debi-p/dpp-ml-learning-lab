const apiBaseInput = document.querySelector("#apiBase");
const modelSelect = document.querySelector("#modelSelect");
const messageInput = document.querySelector("#messageInput");
const correctLabel = document.querySelector("#correctLabel");
const learningRate = document.querySelector("#learningRate");
const trainingInputPanel = document.querySelector("#trainingInputPanel");
const askInputPanel = document.querySelector("#askInputPanel");
const trainingDatasetText = document.querySelector("#trainingDatasetText");
const trainingExpectedCategory = document.querySelector("#trainingExpectedCategory");
const speedControl = document.querySelector("#speedControl");
const speedLabel = document.querySelector("#speedLabel");
const apiStatus = document.querySelector("#apiStatus");

const predictionValue = document.querySelector("#predictionValue");
const vectorShape = document.querySelector("#vectorShape");
const hiddenShape = document.querySelector("#hiddenShape");
const lossValue = document.querySelector("#lossValue");
const cleanedText = document.querySelector("#cleanedText");
const tokenList = document.querySelector("#tokenList");
const vectorMeta = document.querySelector("#vectorMeta");
const featureList = document.querySelector("#featureList");
const hiddenBars = document.querySelector("#hiddenBars");
const activationBars = document.querySelector("#activationBars");
const scoreList = document.querySelector("#scoreList");
const probabilityList = document.querySelector("#probabilityList");
const backpropMeta = document.querySelector("#backpropMeta");
const gradientGrid = document.querySelector("#gradientGrid");
const updateMeta = document.querySelector("#updateMeta");
const beforeWeights = document.querySelector("#beforeWeights");
const afterWeights = document.querySelector("#afterWeights");
const stageTitle = document.querySelector("#stageTitle");
const stageDescription = document.querySelector("#stageDescription");
const flowLane = document.querySelector("#flowLane");
const replayBtn = document.querySelector("#replayBtn");
const previousBtn = document.querySelector("#previousBtn");
const nextBtn = document.querySelector("#nextBtn");
const playBtn = document.querySelector("#playBtn");
const pauseBtn = document.querySelector("#pauseBtn");
const connectionLayer = document.querySelector("#connectionLayer");
const particleLayer = document.querySelector("#particleLayer");
const boxInput = document.querySelector("#boxInput");
const boxAlgorithm = document.querySelector("#boxAlgorithm");
const boxOutput = document.querySelector("#boxOutput");
const flowSvg = document.querySelector("#flowSvg");
const datasetCard = document.querySelector("#datasetCard");
const modeTitle = document.querySelector("#modeTitle");
const modeCard = document.querySelector("#modeCard");
const gitaTracePanel = document.querySelector("#gitaTracePanel");
const gitaTraceTitle = document.querySelector("#gitaTraceTitle");
const gitaSourceList = document.querySelector("#gitaSourceList");
const gitaAnswerText = document.querySelector("#gitaAnswerText");

let lastTrace = null;
let playbackToken = 0;
let currentStages = [];
let currentStepIndex = -1;
let playTimer = null;
let availableModels = [];
let flowPaths = [];
let nodePositions = {};
let activeMode = "training";

const speedOptions = {
  1: { label: "Very Slow", delay: 2200 },
  2: { label: "Slow", delay: 1500 },
  3: { label: "Normal", delay: 950 },
  4: { label: "Fast", delay: 550 },
  5: { label: "Very Fast", delay: 250 },
};

const labelDisplayNames = {
  work: "Work Email",
  personal: "Personal Email",
  promotion: "Promotional Email",
  spam: "Spam Email",
};

const builtInTrainingRow = {
  input: "limited offer claim your reward today",
  correctLabel: "promotion",
};

const stageSectionMap = {
  input_message: "tokens",
  word_tokenization: "tokens",
  vocabulary_lookup: "vector",
  bag_of_words_vectorization: "vector",
  hidden_layer_1: "forward",
  relu_activation: "forward",
  output_layer_2: "scores",
  softmax_probabilities: "softmax",
  loss_calculation: "backprop",
  backpropagation: "backprop",
  weight_update: "update",
  training_step_complete: "update",
  rag_input: "tokens",
  rag_embedding: "vector",
  rag_search: "vector",
  rag_context: "forward",
  rag_answer: "softmax",
  tf_prompt: "tokens",
  tf_tokens: "vector",
  tf_attention: "forward",
  tf_softmax: "softmax",
  tf_generate: "scores",
  rt_question: "tokens",
  rt_retrieve: "vector",
  rt_context: "forward",
  rt_prompt: "forward",
  rt_generate: "scores",
  rt_sources: "softmax",
};

document.querySelector("#inspectForwardBtn").addEventListener("click", () => setModeAndInspect("ask"));
document.querySelector("#inspectTrainingBtn").addEventListener("click", () => setModeAndInspect("training"));
replayBtn.addEventListener("click", () => {
  if (lastTrace) playTrace(lastTrace);
});
previousBtn.addEventListener("click", previousStep);
nextBtn.addEventListener("click", nextStep);
playBtn.addEventListener("click", playFromCurrentStep);
pauseBtn.addEventListener("click", pausePlayback);
speedControl.addEventListener("input", () => {
  const option = speedOptions[speedControl.value] || speedOptions[3];
  speedLabel.textContent = option.label;
});
apiBaseInput.addEventListener("change", loadModels);

resetDisplay();
syncModeControls();
loadModels();

async function loadModels() {
  setStatus("Checking API", "");
  try {
    const response = await fetch(`${apiBaseInput.value}/models`);
    if (!response.ok) throw new Error(`API returned ${response.status}`);
    const models = await response.json();
    availableModels = models.filter(isVisualizableModel);
    modelSelect.innerHTML = "";
    for (const model of availableModels) {
      const option = document.createElement("option");
      option.value = model.model_id;
      option.textContent = model.model_id;
      modelSelect.append(option);
    }
    updateCorrectLabelControl();
    setStatus("API connected", "ok");
    activeMode = "training";
    syncModeControls();
    await inspect("training");
  } catch (error) {
    setStatus("API unavailable", "error");
    modelSelect.innerHTML = `<option>dpp-email-classifier-small-v1</option>`;
    correctLabel.innerHTML = `<option value="work">Work Email</option><option value="personal">Personal Email</option><option value="promotion">Promotional Email</option><option value="spam">Spam Email</option>`;
  }
}

modelSelect.addEventListener("change", updateCorrectLabelControl);
modelSelect.addEventListener("change", () => {
  const model = selectedModel();
  activeMode = supportsTraining(model) ? "training" : "ask";
  messageInput.value = defaultPrompt(model);
  syncModeControls();
  inspect(activeMode === "training" ? "training" : "forward");
});

function updateCorrectLabelControl() {
  const model = selectedModel();
  correctLabel.innerHTML = "";
  if (!model || model.target_type !== "class_label" || !Array.isArray(model.labels)) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No class labels";
    correctLabel.append(option);
    correctLabel.disabled = true;
    return;
  }

  correctLabel.disabled = activeMode === "ask";
  for (const label of model.labels) {
    const option = document.createElement("option");
    option.value = label;
    option.textContent = displayLabel(label);
    correctLabel.append(option);
  }
}

function displayLabel(label) {
  return labelDisplayNames[label] || label;
}

function setModeAndInspect(mode) {
  if (mode === "training" && !supportsTraining(selectedModel())) {
    activeMode = "ask";
    syncModeControls();
    inspect("forward");
    return;
  }
  activeMode = mode;
  syncModeControls();
  inspect(mode === "training" ? "training" : "forward");
}

function syncModeControls() {
  const model = selectedModel();
  const trainingSupported = supportsTraining(model);
  document.querySelector("#inspectTrainingBtn").hidden = !trainingSupported;
  document.querySelector("#inspectTrainingBtn").classList.toggle("active", activeMode === "training");
  document.querySelector("#inspectForwardBtn").classList.toggle("active", activeMode === "ask");
  trainingInputPanel.hidden = true;
  askInputPanel.hidden = activeMode !== "ask";
  messageInput.placeholder = promptPlaceholder(model);
  if (activeMode === "ask" && !messageInput.value.trim()) messageInput.value = defaultPrompt(model);
  if (trainingDatasetText) trainingDatasetText.textContent = builtInTrainingRow.input;
  if (trainingExpectedCategory) trainingExpectedCategory.textContent = `Expected: ${displayLabel(builtInTrainingRow.correctLabel)}`;
  correctLabel.value = builtInTrainingRow.correctLabel;
  correctLabel.disabled = true;
  learningRate.disabled = activeMode === "ask";
  document.querySelectorAll(".training-only").forEach((section) => {
    section.hidden = activeMode === "ask";
  });
  datasetCard.textContent = datasetText(model);
  modeTitle.textContent = activeMode === "training" ? "Training Mode" : "Ask Mode";
  modeCard.textContent = modeText(model, trainingSupported);
}

async function inspect(mode) {
  const buttons = [...document.querySelectorAll("button")];
  buttons.forEach((button) => (button.disabled = true));
  try {
    const model = selectedModel();
    const endpoint = endpointFor(model, mode);
    const payload = {
      model_id: modelSelect.value,
      input: mode === "training" && supportsTraining(model) ? builtInTrainingRow.input : messageInput.value,
    };
    if (mode === "training" && supportsTraining(model)) {
      payload.correct_label = builtInTrainingRow.correctLabel;
      payload.learning_rate = Number(learningRate.value);
    }

    const response = await fetch(`${apiBaseInput.value}/${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const rawTrace = await response.json();
    if (!response.ok) throw new Error(rawTrace.detail || `API returned ${response.status}`);
    const trace = normalizeTrace(rawTrace, model);
    lastTrace = trace;
    loadTrace(trace);
    nextStep();
    setStatus("Trace loaded", "ok");
  } catch (error) {
    setStatus(error.message, "error");
  } finally {
    buttons.forEach((button) => (button.disabled = false));
    updatePlaybackButtons();
  }
}

async function playTrace(trace) {
  loadTrace(trace);
  playFromCurrentStep();
}

function loadTrace(trace) {
  pausePlayback();
  playbackToken += 1;
  resetDisplay();
  drawFlowScene(trace);
  currentStages = buildStages(trace);
  currentStepIndex = -1;
  updatePlaybackButtons();
}

function previousStep() {
  if (!currentStages.length) return;
  pausePlayback();
  currentStepIndex = Math.max(0, currentStepIndex - 1);
  renderCurrentStep();
}

function nextStep() {
  if (!currentStages.length) return;
  pausePlayback();
  currentStepIndex = Math.min(currentStages.length - 1, currentStepIndex + 1);
  renderCurrentStep();
}

function playFromCurrentStep() {
  if (!currentStages.length) return;
  pausePlayback();
  const token = ++playbackToken;
  if (currentStepIndex < 0) currentStepIndex = 0;
  renderCurrentStep();
  playTimer = window.setTimeout(() => autoAdvance(token), getStageDelay(currentStages[currentStepIndex]));
  updatePlaybackButtons();
}

function autoAdvance(token) {
  if (token !== playbackToken) return;
  if (currentStepIndex >= currentStages.length - 1) {
    pausePlayback();
    return;
  }
  currentStepIndex += 1;
  renderCurrentStep();
  playTimer = window.setTimeout(() => autoAdvance(token), getStageDelay(currentStages[currentStepIndex]));
}

function pausePlayback() {
  if (playTimer) {
    window.clearTimeout(playTimer);
    playTimer = null;
  }
  updatePlaybackButtons();
}

function renderCurrentStep() {
  const stage = currentStages[currentStepIndex];
  if (!stage) return;
  setStage(stage, currentStepIndex, currentStages.length);
  stage.render();
  updatePlaybackButtons();
}

function updatePlaybackButtons() {
  const hasTrace = currentStages.length > 0;
  previousBtn.disabled = !hasTrace || currentStepIndex <= 0;
  nextBtn.disabled = !hasTrace || currentStepIndex >= currentStages.length - 1;
  playBtn.disabled = !hasTrace || Boolean(playTimer);
  pauseBtn.disabled = !playTimer;
  replayBtn.disabled = !lastTrace;
}

function getStageDelay(stage) {
  const option = speedOptions[speedControl.value] || speedOptions[3];
  const baseDelay = option.delay;
  if (stage.quick) {
    return Math.max(280, Math.round(baseDelay * 0.42));
  }
  return baseDelay;
}

function buildStages(trace) {
  const nodesById = Object.fromEntries((trace.flow?.nodes || []).map((node) => [node.id, node]));
  return (trace.flow?.timeline || []).map((nodeId) => {
    const node = nodesById[nodeId];
    return {
      id: node.id,
      title: node.name,
      description: node.description,
      quick: ["output_layer_2", "softmax_probabilities"].includes(node.id),
      node,
      detail: () => ({
        input: formatNodePort(node.input),
        algorithm: `${node.algorithm.label}\n${formatPreview(node.algorithm.value)}`,
        output: formatNodePort(node.output),
      }),
      render: () => renderGenericStage(trace, node),
    };
  });
}

function selectedModel() {
  return availableModels.find((item) => item.model_id === modelSelect.value) || { model_id: modelSelect.value };
}

function supportsTraining(model) {
  return Boolean(model?.supports?.includes("inspect_training_step"));
}

function isVisualizableModel(model) {
  const supports = model?.supports || [];
  return [
    "inspect_training_step",
    "inspect_forward",
    "inspect_rag",
    "inspect_transformer",
    "inspect_rag_transformer",
  ].some((capability) => supports.includes(capability));
}

function promptPlaceholder(model) {
  if (model?.model_id === "dpp-gita-rag-transformer-v1") return "Ask a Gita question for retrieval + transformer generation";
  if (model?.model_id === "dpp-gita-rag-assistant-v2") return "Ask a Gita question for source-backed retrieval";
  if (model?.model_id === "dpp-gita-tiny-transformer-v1") return "Enter a short prompt for next-token generation";
  return "Type a message to classify";
}

function defaultPrompt(model) {
  if (model?.model_id?.includes("gita")) return "How can I control anger?";
  return "Can we review the project deadline tomorrow?";
}

function endpointFor(model, mode) {
  if (mode === "training" && supportsTraining(model)) return "inspect-training-step";
  if (model?.supports?.includes("inspect_rag_transformer")) return "inspect-rag-transformer";
  if (model?.supports?.includes("inspect_rag")) return "inspect-rag";
  if (model?.supports?.includes("inspect_transformer")) return "inspect-transformer";
  return "inspect-forward";
}

function datasetText(model) {
  if (model?.model_id === "dpp-gita-rag-transformer-v1") {
    return "Input: user Gita question\nRetriever: dpp-gita-embedding-small-v1\nGenerator: dpp-gita-tiny-transformer-v1\nOutput: experimental generated answer + sources";
  }
  if (model?.model_id === "dpp-gita-rag-assistant-v2") {
    return "Input: user Gita question\nSource: parsed Bhagavad Gita verses\nRetrieval: neural embedding cosine search\nOutput: source-backed answer";
  }
  if (model?.model_id === "dpp-gita-tiny-transformer-v1") {
    return "Input: prompt text\nSource: Gita + Q&A training text\nTask: next-token prediction\nOutput: generated tokens";
  }
  return "Input: built-in email dataset\nSource: spam and ham emails\nOutput: expected category\nCategories: Spam Email, Personal Email, Work Email, Promotional Email";
}

function modeText(model, trainingSupported) {
  if (!model?.model_id?.includes("gita")) {
    return "Each row flows through the model\nModel predicts an email category\nLoss checks prediction vs expected category\nBackpropagation updates weights for training rows";
  }
  if (trainingSupported) return "Training and ask mode are available.";
  return "Ask Mode only\nThe model receives your question or prompt\nThe visualizer shows retrieval, context, attention/generation, and final output depending on selected model";
}

function normalizeTrace(trace, model) {
  if (trace?.flow) {
    trace.trace_type = "email";
    return trace;
  }
  if (model?.supports?.includes("inspect_rag_transformer")) return normalizeRagTransformerTrace(trace);
  if (model?.supports?.includes("inspect_rag")) return normalizeRagTrace(trace);
  if (model?.supports?.includes("inspect_transformer")) return normalizeTransformerTrace(trace);
  return trace;
}

function normalizeRagTrace(trace) {
  const nodes = [
    flowNode("rag_input", "User Question", "raw question", "question text", "Receive question", trace.input?.text, "question", trace.input?.text),
    flowNode("rag_embedding", "Question Embedding", "64 dims", "question tokens", "Neural embedding model", trace.input?.text, "query vector", trace.retrieval?.embedding_shape),
    flowNode("rag_search", "Similarity Search", "top verses", "query vector", trace.retrieval?.algorithm, trace.retrieval?.embedding_shape, "top results", topReferences(trace.retrieval?.results)),
    flowNode("rag_context", "Augmented Context", "sources", "top verses", "Context builder", topReferences(trace.retrieval?.results), "context", trace.augmented_context?.sources),
    flowNode("rag_answer", "RAG Answer", "source-backed", "context", "Rule-based answer builder", trace.augmented_context, "answer", trace.answer?.answer),
  ];
  return withFlow(trace, "rag", nodes, lineEdges(nodes), {
    prediction: "RAG answer",
    vectorShape: (trace.retrieval?.embedding_shape || []).join(" x "),
    hiddenShape: "top-k sources",
    loss: "-",
  });
}

function normalizeTransformerTrace(trace) {
  const nodes = [
    flowNode("tf_prompt", "Prompt", "raw text", "prompt", "Receive prompt", trace.input?.prompt, "prompt", trace.input?.prompt),
    flowNode("tf_tokens", "Tokenization", "tokens", "prompt", "Tokenizer + vocabulary lookup", trace.input?.prompt, "token ids", trace.tokenization),
    flowNode("tf_attention", "Causal Attention", "context window", "token ids", "Token/position embeddings + attention", trace.tokenization?.token_ids, "next-token scores", firstStep(trace)?.top_tokens),
    flowNode("tf_softmax", "Softmax", "probabilities", "logits", "Softmax over vocabulary", firstStep(trace)?.top_tokens, "chosen token", firstStep(trace)?.next_token),
    flowNode("tf_generate", "Generated Tokens", "repeat", "previous tokens", "Sample next token repeatedly", trace.generation_steps, "generated text", trace.generated_text),
  ];
  return withFlow(trace, "transformer", nodes, lineEdges(nodes), {
    prediction: "Generated text",
    vectorShape: `${trace.tokenization?.vocabulary_size || "-"} vocab`,
    hiddenShape: `${trace.tokenization?.context_length || "-"} context`,
    loss: "-",
  });
}

function normalizeRagTransformerTrace(trace) {
  const nodes = [
    flowNode("rt_question", "User Question", "raw question", "question", "Receive question", trace.input?.question, "question", trace.input?.question),
    flowNode("rt_retrieve", "Embedding Retrieval", "top verses", "question", trace.retrieval?.algorithm, trace.input?.question, "sources", topReferences(trace.retrieval?.results)),
    flowNode("rt_context", "RAG Context", "augmented", "top verses", "Context builder", trace.augmented_context?.sources, "prompt context", trace.transformer_prompt),
    flowNode("rt_prompt", "Transformer Prompt", "question + context", "augmented context", "Prompt builder", trace.transformer_prompt, "token prompt", trace.transformer_prompt),
    flowNode("rt_generate", "Tiny Transformer", "next tokens", "prompt tokens", "Causal attention + softmax sampling", trace.generation_steps, "generated answer", trace.answer),
    flowNode("rt_sources", "Answer + Sources", "final", "generated answer", "Attach source citations", trace.answer, "sources", trace.sources),
  ];
  return withFlow(trace, "rag_transformer", nodes, lineEdges(nodes), {
    prediction: "RAG + transformer",
    vectorShape: "retrieval + prompt",
    hiddenShape: `${trace.transformer_config?.context_length || "-"} context`,
    loss: "-",
  });
}

function flowNode(id, name, subtitle, inputLabel, algorithmLabel, inputPreview, outputLabel, outputPreview) {
  return {
    id,
    name,
    subtitle,
    description: `${name}: ${subtitle}`,
    algorithm: { label: algorithmLabel || name, value: "" },
    input: { label: inputLabel, shape: inferShape(inputPreview), preview: inputPreview },
    output: { label: outputLabel, shape: inferShape(outputPreview), preview: outputPreview },
  };
}

function withFlow(trace, traceType, nodes, edges, summary) {
  return {
    ...trace,
    trace_type: traceType,
    flow: { nodes, edges, timeline: nodes.map((node) => node.id) },
    summary,
  };
}

function lineEdges(nodes) {
  return nodes.slice(1).map((node, index) => ({
    from: nodes[index].id,
    to: node.id,
    direction: "forward",
  }));
}

function inferShape(value) {
  if (Array.isArray(value)) return String(value.length);
  if (value && typeof value === "object") return Object.keys(value).length ? "object" : "-";
  if (typeof value === "string") return `${value.split(/\s+/).filter(Boolean).length} words`;
  return "-";
}

function topReferences(results = []) {
  return results.slice(0, 5).map((item) => `Chapter ${item.chapter}, Verse ${item.verse} (${item.score})`);
}

function firstStep(trace) {
  return (trace.generation_steps || [])[0] || {};
}

function renderTrace(trace) {
  predictionValue.textContent = trace.prediction || "Unknown";
  vectorShape.textContent = trace.vectorization?.shape || "-";
  hiddenShape.textContent = trace.forward_propagation?.hidden_layer?.shape || "-";
  lossValue.textContent = trace.training ? trace.training.loss.toFixed(4) : "-";

  cleanedText.textContent = trace.input.cleaned;
  renderTokens(trace.input.tokens || []);
  renderFeatures(trace.vectorization);
  renderBars(hiddenBars, trace.forward_propagation.hidden_layer.values, "z");
  renderBars(activationBars, trace.forward_propagation.activation.values, "a");
  renderScores(trace.forward_propagation.output_layer.scores);
  renderProbabilities(trace.forward_propagation.softmax.probabilities);
  renderBackprop(trace.backpropagation);
  renderWeightUpdate(trace.weight_update);
}

function renderInputStage(trace) {
  cleanedText.textContent = trace.input.cleaned;
  renderTokens(trace.input.tokens || []);
}

function renderGenericStage(trace, node) {
  if (trace.trace_type && trace.trace_type !== "email") {
    renderGitaStage(trace, node);
    return;
  }
  const legacyRenderers = {
    input_message: () => {
      cleanedText.textContent = trace.input.cleaned;
    },
    word_tokenization: () => renderInputStage(trace),
    vocabulary_lookup: () => renderVocabularyStage(trace),
    bag_of_words_vectorization: () => renderVectorStage(trace),
    hidden_layer_1: () => renderHiddenStage(trace),
    relu_activation: () => renderReluStage(trace),
    output_layer_2: () => renderScoreStage(trace),
    softmax_probabilities: () => renderSoftmaxStage(trace),
    loss_calculation: () => renderLossStage(trace),
    backpropagation: () => renderBackpropStage(trace),
    weight_update: () => renderUpdateStage(trace),
  };
  const renderer = legacyRenderers[node.id];
  if (renderer) renderer();
}

function renderGitaStage(trace, node) {
  predictionValue.textContent = trace.summary?.prediction || trace.model_id || "Gita model";
  vectorShape.textContent = trace.summary?.vectorShape || "-";
  hiddenShape.textContent = trace.summary?.hiddenShape || "-";
  lossValue.textContent = trace.summary?.loss || "-";
  cleanedText.textContent = trace.input?.question || trace.input?.text || trace.input?.prompt || "";
  renderTokens(extractTraceTokens(trace));
  renderTraceFeatures(trace);
  renderGitaPanel(trace);
  renderGenerationProbabilities(trace);
}

function extractTraceTokens(trace) {
  if (trace.tokenization?.prompt_tokens) return trace.tokenization.prompt_tokens;
  const text = trace.input?.question || trace.input?.text || trace.input?.prompt || "";
  return String(text).toLowerCase().split(/\s+/).filter(Boolean).slice(0, 24);
}

function renderTraceFeatures(trace) {
  featureList.innerHTML = "";
  if (trace.retrieval?.results) {
    vectorMeta.textContent = `Retrieval results from ${trace.retrieval.algorithm || "similarity search"}.`;
    trace.retrieval.results.slice(0, 5).forEach((result) => {
      featureList.append(chip(`Ch ${result.chapter}.${result.verse}: ${Number(result.score).toFixed(3)}`, "feature"));
    });
    return;
  }
  if (trace.tokenization) {
    vectorMeta.textContent = `Vocabulary size: ${trace.tokenization.vocabulary_size}. Context length: ${trace.tokenization.context_length}.`;
    (trace.tokenization.token_ids || []).slice(0, 24).forEach((id, index) => {
      featureList.append(chip(`t${index + 1}: ${id}`, "feature"));
    });
    return;
  }
  vectorMeta.textContent = "Trace features appear here.";
}

function renderGitaPanel(trace) {
  gitaTracePanel.hidden = false;
  gitaTraceTitle.textContent = trace.model_id || "Gita Trace";
  gitaSourceList.innerHTML = "";
  const sources = trace.sources || trace.augmented_context?.sources || trace.answer?.sources || [];
  sources.slice(0, 5).forEach((source) => {
    const item = document.createElement("div");
    item.className = "source-item";
    const reference = source.reference || `Chapter ${source.chapter}, Verse ${source.verse}`;
    const score = source.score !== undefined ? `score ${Number(source.score).toFixed(3)}` : "";
    item.innerHTML = `<strong>${reference}</strong><span>${score}</span>`;
    gitaSourceList.append(item);
  });
  if (!sources.length) {
    gitaSourceList.append(chip("No retrieved sources for this trace", "feature"));
  }
  gitaAnswerText.textContent = trace.answer?.answer || trace.answer || trace.generated_text || "Waiting for model output...";
}

function renderGenerationProbabilities(trace) {
  scoreList.innerHTML = "";
  probabilityList.innerHTML = "";
  const step = firstStep(trace);
  const topTokens = step.top_tokens || [];
  if (!topTokens.length) return;
  const scores = {};
  const probs = {};
  topTokens.slice(0, 8).forEach((item) => {
    scores[item.token] = item.probability;
    probs[item.token] = item.probability;
  });
  renderMetricRows(scoreList, scores, true);
  renderMetricRows(probabilityList, probs, true);
}

function renderVectorStage(trace) {
  vectorShape.textContent = trace.vectorization?.shape || "-";
  renderFeatures(trace.vectorization);
}

function renderVocabularyStage(trace) {
  vectorMeta.textContent = `Vocabulary size: ${trace.vectorization.vocab_size}. Matching known tokens before vectorization.`;
  renderFeatures(trace.vectorization);
}

function renderHiddenStage(trace) {
  hiddenShape.textContent = trace.forward_propagation?.hidden_layer?.shape || "-";
  renderBars(hiddenBars, trace.forward_propagation.hidden_layer.values, "z");
}

function renderReluStage(trace) {
  renderBars(activationBars, trace.forward_propagation.activation.values, "a");
}

function renderScoreStage(trace) {
  renderScores(trace.forward_propagation.output_layer.scores);
}

function renderSoftmaxStage(trace) {
  predictionValue.textContent = trace.prediction || "Unknown";
  renderProbabilities(trace.forward_propagation.softmax.probabilities);
}

function renderBackpropStage(trace) {
  renderBackprop(trace.backpropagation);
}

function renderLossStage(trace) {
  lossValue.textContent = trace.training ? trace.training.loss.toFixed(4) : "-";
}

function renderUpdateStage(trace) {
  renderWeightUpdate(trace.weight_update);
}

function setStage(stage, index, total) {
  stageTitle.textContent = stage.title;
  stageDescription.textContent = stage.description;
  renderBoxDetail(stage.detail ? stage.detail() : null);
  flowLane.style.setProperty("--flow-progress", `${(index / Math.max(1, total - 1)) * 88}%`);
  const activeSection = stageSectionMap[stage.id] || stage.id;

  [...flowLane.children].forEach((item, itemIndex) => {
    item.classList.toggle("active", itemIndex <= index);
  });

  setFlowSceneStage(stage.id);

  document.querySelectorAll(".step").forEach((step) => {
    const isActive = step.dataset.stage === activeSection;
    step.classList.toggle("active", isActive);
    step.classList.toggle("visible", step.classList.contains("visible") || isActive);
  });
}

function resetDisplay() {
  predictionValue.textContent = "Waiting";
  vectorShape.textContent = "-";
  hiddenShape.textContent = "-";
  lossValue.textContent = "-";
  cleanedText.textContent = "Cleaned text appears here.";
  tokenList.innerHTML = "";
  vectorMeta.textContent = "Bag-of-Words features appear here.";
  featureList.innerHTML = "";
  hiddenBars.innerHTML = "";
  activationBars.innerHTML = "";
  scoreList.innerHTML = "";
  probabilityList.innerHTML = "";
  backpropMeta.textContent = "Run training-step inspection to see gradients.";
  gradientGrid.innerHTML = "";
  updateMeta.textContent = "Run training-step inspection to see before and after values.";
  beforeWeights.textContent = "{}";
  afterWeights.textContent = "{}";
  stageTitle.textContent = "Waiting for inspection";
  stageDescription.textContent = "Run a forward pass or training step to watch the model process data.";
  boxInput.textContent = "Waiting for data...";
  boxAlgorithm.textContent = "Waiting for data...";
  boxOutput.textContent = "Waiting for data...";
  gitaTracePanel.hidden = true;
  gitaSourceList.innerHTML = "";
  gitaAnswerText.textContent = "Waiting for model output...";
  flowLane.style.setProperty("--flow-progress", "0%");
  [...flowLane.children].forEach((item) => item.classList.remove("active"));
  document.querySelectorAll(".step").forEach((step) => {
    step.classList.remove("active", "visible");
  });
  document.querySelectorAll(".training-only").forEach((section) => {
    section.hidden = activeMode === "ask";
  });
  document.querySelectorAll(".flow-node, .flow-line").forEach((item) => {
    item.classList.remove("active", "completed");
  });
  particleLayer.innerHTML = "";
}

function drawFlowScene(trace) {
  connectionLayer.innerHTML = "";
  particleLayer.innerHTML = "";
  removeFlowNodes();
  if (!trace?.flow) return;

  const nodeLayer = ensureNodeLayer();
  const nodes = trace.flow.nodes || [];
  const edges = trace.flow.edges || [];
  renderFlowLane(nodes, trace.flow.timeline || []);
  nodePositions = calculateNodePositions(nodes);
  flowPaths = edges.map((edge) => ({
    ...edge,
    stage: edge.direction === "forward" ? edge.to : edge.from,
    backward: edge.direction !== "forward",
    curve: buildCurve(edge, nodePositions),
  }));

  for (const path of flowPaths) {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "path");
    line.setAttribute("d", path.curve);
    line.setAttribute("class", `flow-line ${path.backward ? "backward" : "forward"}`);
    line.dataset.stage = path.stage;
    connectionLayer.append(line);
  }

  for (const node of nodes) {
    const position = nodePositions[node.id];
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", "flow-node dynamic-flow-node");
    group.dataset.flowNode = node.id;
    group.setAttribute("transform", `translate(${position.x} ${position.y})`);

    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("width", String(position.width));
    rect.setAttribute("height", String(position.height));
    rect.setAttribute("rx", "14");

    const title = createWrappedSvgText({
      text: node.name,
      x: position.width / 2,
      y: 25,
      maxCharsPerLine: 15,
      className: "node-title",
    });

    const subtitle = document.createElementNS("http://www.w3.org/2000/svg", "text");
    subtitle.setAttribute("x", String(position.width / 2));
    subtitle.setAttribute("y", String(position.height - 18));
    subtitle.setAttribute("class", "node-subtitle");
    subtitle.textContent = node.subtitle;

    group.append(rect, title, subtitle);
    nodeLayer.append(group);
  }
}

function createWrappedSvgText({ text, x, y, maxCharsPerLine, className }) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", "text");
  element.setAttribute("x", String(x));
  element.setAttribute("y", String(y));
  element.setAttribute("class", className);

  const lines = wrapWords(text, maxCharsPerLine).slice(0, 3);
  lines.forEach((line, index) => {
    const tspan = document.createElementNS("http://www.w3.org/2000/svg", "tspan");
    tspan.setAttribute("x", String(x));
    tspan.setAttribute("dy", index === 0 ? "0" : "17");
    tspan.textContent = line;
    element.append(tspan);
  });
  return element;
}

function wrapWords(text, maxCharsPerLine) {
  const words = String(text).split(/\s+/).filter(Boolean);
  const lines = [];
  let current = "";

  for (const word of words) {
    const next = current ? `${current} ${word}` : word;
    if (next.length <= maxCharsPerLine) {
      current = next;
      continue;
    }
    if (current) lines.push(current);
    current = word.length > maxCharsPerLine ? `${word.slice(0, maxCharsPerLine - 1)}…` : word;
  }

  if (current) lines.push(current);
  return lines.length ? lines : [String(text)];
}

function removeFlowNodes() {
  document.querySelectorAll(".dynamic-flow-node").forEach((node) => node.remove());
  document.querySelectorAll("#flowSvg > .flow-node").forEach((node) => node.remove());
}

function ensureNodeLayer() {
  let nodeLayer = document.querySelector("#nodeLayer");
  if (!nodeLayer) {
    nodeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
    nodeLayer.setAttribute("id", "nodeLayer");
    flowSvg.append(nodeLayer);
  }
  return nodeLayer;
}

function calculateNodePositions(nodes) {
  const positions = {};
  const width = 170;
  const height = 92;
  const manualPositions = {
    input_message: { x: 36, y: 110 },
    word_tokenization: { x: 218, y: 110 },
    vocabulary_lookup: { x: 400, y: 110 },
    bag_of_words_vectorization: { x: 582, y: 110 },
    hidden_layer_1: { x: 764, y: 110 },
    relu_activation: { x: 946, y: 110 },
    output_layer_2: { x: 1128, y: 110 },
    softmax_probabilities: { x: 1310, y: 110 },
    loss_calculation: { x: 1310, y: 250 },
    backpropagation: { x: 1310, y: 360 },
    weight_update: { x: 1310, y: 470 },
    training_step_complete: { x: 1038, y: 470 },
    rag_input: { x: 36, y: 110 },
    rag_embedding: { x: 308, y: 110 },
    rag_search: { x: 580, y: 110 },
    rag_context: { x: 852, y: 110 },
    rag_answer: { x: 1124, y: 110 },
    tf_prompt: { x: 36, y: 110 },
    tf_tokens: { x: 308, y: 110 },
    tf_attention: { x: 580, y: 110 },
    tf_softmax: { x: 852, y: 110 },
    tf_generate: { x: 1124, y: 110 },
    rt_question: { x: 36, y: 110 },
    rt_retrieve: { x: 288, y: 110 },
    rt_context: { x: 540, y: 110 },
    rt_prompt: { x: 792, y: 110 },
    rt_generate: { x: 1044, y: 110 },
    rt_sources: { x: 1296, y: 110 },
  };

  nodes.forEach((node, index) => {
    const manual = manualPositions[node.id] || { x: 42 + (index % 6) * 243, y: index < 6 ? 90 : 300 };
    positions[node.id] = { ...manual, width, height };
  });
  return positions;
}

function renderFlowLane(nodes, timeline) {
  const nodesById = Object.fromEntries(nodes.map((node) => [node.id, node]));
  flowLane.innerHTML = "";
  for (const nodeId of timeline) {
    const node = nodesById[nodeId];
    if (!node) continue;
    const item = document.createElement("span");
    item.dataset.flowNode = node.id;
    item.textContent = node.name.replace("Bag-of-Words ", "").replace(" Probabilities", "");
    flowLane.append(item);
  }
}

function buildCurve(edge, positions) {
  const from = positions[edge.from];
  const to = positions[edge.to];
  if (!from || !to) return "M 0 0";

  const x1 = from.x + from.width;
  const y1 = from.y + from.height / 2;
  const x2 = to.x;
  const y2 = to.y + to.height / 2;
  const direction = edge.direction || "forward";

  if (direction === "forward") {
    const isDrop = to.y > from.y + 80;

    if (isDrop) {
      const fromBottomX = from.x + from.width / 2;
      const fromBottomY = from.y + from.height;
      const toTopX = to.x + to.width / 2;
      const toTopY = to.y;
      const midY = (fromBottomY + toTopY) / 2;
      return `M ${fromBottomX} ${fromBottomY} C ${fromBottomX} ${midY}, ${toTopX} ${midY}, ${toTopX} ${toTopY}`;
    }

    const mid = (x1 + x2) / 2;
    return `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`;
  }

  const fromBottomX = from.x + from.width / 2;
  const fromBottomY = from.y + from.height;
  const toTopX = to.x + to.width / 2;
  const toTopY = to.y;
  const midY = (fromBottomY + toTopY) / 2;
  return `M ${fromBottomX} ${fromBottomY} C ${fromBottomX} ${midY}, ${toTopX} ${midY}, ${toTopX} ${toTopY}`;
}

function setFlowSceneStage(stageId) {
  document.querySelectorAll(".flow-node").forEach((node) => {
    const isActive = node.dataset.flowNode === stageId;
    node.classList.toggle("active", isActive);
    node.classList.toggle("completed", node.classList.contains("completed") || isActive);
  });
  document.querySelectorAll(".flow-line").forEach((line) => {
    line.classList.toggle("active", line.dataset.stage === stageId);
  });
  spawnParticles(stageId);
}

function spawnParticles(stageId) {
  particleLayer.innerHTML = "";
  const activePaths = flowPaths.filter((path) => path.stage === stageId);
  activePaths.forEach((path, index) => {
    for (let particleIndex = 0; particleIndex < 4; particleIndex += 1) {
      const particle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
      const id = `motion-${stageId}-${index}-${particleIndex}-${Date.now()}`;
      const motionPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
      motionPath.setAttribute("id", id);
      motionPath.setAttribute("d", path.curve);
      motionPath.setAttribute("fill", "none");
      particle.setAttribute("r", String(path.backward ? 4 : 3.5));
      particle.setAttribute("class", `flow-particle ${path.backward ? "backward" : "forward"}`);

      const animateMotion = document.createElementNS("http://www.w3.org/2000/svg", "animateMotion");
      const option = speedOptions[speedControl.value] || speedOptions[3];
      animateMotion.setAttribute("dur", `${Math.max(0.8, option.delay / 900)}s`);
      animateMotion.setAttribute("begin", `${particleIndex * 0.18}s`);
      animateMotion.setAttribute("repeatCount", "indefinite");

      const mpath = document.createElementNS("http://www.w3.org/2000/svg", "mpath");
      mpath.setAttributeNS("http://www.w3.org/1999/xlink", "href", `#${id}`);
      animateMotion.append(mpath);
      particle.append(animateMotion);

      particleLayer.append(motionPath, particle);
    }
  });
}

function renderBoxDetail(detail) {
  if (!detail) return;
  boxInput.textContent = detail.input;
  boxAlgorithm.textContent = detail.algorithm;
  boxOutput.textContent = detail.output;
}

function formatNodePort(port) {
  return `${port.label}\nshape: ${port.shape}\n${formatPreview(port.preview)}`;
}

function formatPreview(value) {
  if (typeof value === "string") return value;
  return JSON.stringify(value, null, 2);
}

function renderTokens(tokens) {
  tokenList.innerHTML = "";
  for (const token of tokens) {
    tokenList.append(chip(token, "token"));
  }
}

function renderFeatures(vectorization) {
  featureList.innerHTML = "";
  const features = vectorization.non_zero_features || {};
  vectorMeta.textContent = `${vectorization.algorithm}. Vocabulary size: ${vectorization.vocab_size}. Showing non-zero features only.`;
  for (const [word, count] of Object.entries(features)) {
    const item = document.createElement("div");
    item.className = "feature";
    item.innerHTML = `${word}: <strong>${count}</strong>`;
    featureList.append(item);
  }
}

function renderBars(container, values, prefix) {
  container.innerHTML = "";
  const max = Math.max(0.001, ...values.map((value) => Math.abs(value)));
  values.forEach((value, index) => {
    const row = document.createElement("div");
    row.className = "bar-item";
    const width = Math.max(2, Math.abs(value / max) * 100);
    const tone = value < 0 ? "negative" : "positive";
    row.innerHTML = `
      <span>${prefix}${index + 1}</span>
      <div class="bar-track"><div class="bar-fill ${tone}" style="width:${width}%"></div></div>
      <span>${value.toFixed(3)}</span>
    `;
    container.append(row);
  });
}

function renderScores(scores) {
  scoreList.innerHTML = "";
  renderMetricRows(scoreList, scores, false);
}

function renderProbabilities(probabilities) {
  probabilityList.innerHTML = "";
  renderMetricRows(probabilityList, probabilities, true);
}

function renderMetricRows(container, values, percent) {
  const max = Math.max(0.001, ...Object.values(values).map((value) => Math.abs(value)));
  for (const [label, value] of Object.entries(values)) {
    const row = document.createElement("div");
    row.className = percent ? "probability-row" : "score-row";
    const display = percent ? `${(value * 100).toFixed(1)}%` : value.toFixed(3);
    row.innerHTML = `
      <span>${displayLabel(label)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${Math.max(2, Math.abs(value / max) * 100)}%"></div></div>
      <span>${display}</span>
    `;
    container.append(row);
  }
}

function renderBackprop(backpropagation) {
  gradientGrid.innerHTML = "";
  if (!backpropagation) {
    backpropMeta.textContent = "Run training-step inspection to see gradients.";
    return;
  }
  backpropMeta.textContent = backpropagation.algorithm;
  for (const [name, summary] of Object.entries(backpropagation)) {
    if (name === "algorithm") continue;
    gradientGrid.append(chip(`${name}: ${summary.shape}`, "gradient"));
  }
}

function renderWeightUpdate(weightUpdate) {
  if (!weightUpdate) {
    updateMeta.textContent = "Run training-step inspection to see before and after values.";
    beforeWeights.textContent = "{}";
    afterWeights.textContent = "{}";
    return;
  }
  updateMeta.textContent = weightUpdate.algorithm;
  beforeWeights.textContent = JSON.stringify(weightUpdate.before, null, 2);
  afterWeights.textContent = JSON.stringify(weightUpdate.after, null, 2);
}

function chip(text, className) {
  const element = document.createElement("span");
  element.className = className;
  element.textContent = text;
  return element;
}

function setStatus(message, kind) {
  apiStatus.textContent = message;
  apiStatus.className = `api-status ${kind}`;
}

function wait(ms) {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}
