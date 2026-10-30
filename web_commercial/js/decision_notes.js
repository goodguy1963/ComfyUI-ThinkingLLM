import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

const DECISION_NODE_NAMES = new Set([
  "ThinkingLLM_DecisionRouter",
  "ThinkingLLM_DecisionRouterText",
  "ThinkingLLM_DecisionFieldsImage",
  "ThinkingLLM_DecisionFieldsText",
]);
const NOTES_WIDGET_NAME = "recommended_settings";

function buildDecisionNotes(node) {
  const isFieldNode = node.comfyClass.includes("DecisionFields");
  const inputKind = node.comfyClass.endsWith("Text") ? "text" : "image";
  if (isFieldNode) {
    return [
      `Decision Fields (${inputKind})`,
      "BOOLEAN: ask a yes/no question. Fields are independent, so multiple answers can be true.",
      "Threshold (0-1): minimum probability for true; 0.50 is the default.",
      "INT: choose a whole number from Minimum through Maximum, inclusive. Useful for counts.",
      "Outside that range, the integer is clamped to its nearest bound and marked in decision_json.",
      "decision_json: detailed machine-readable values and probabilities; decision: readable multiline summary.",
      "Typed field sockets return native Boolean or Integer batches in input order.",
      "Backend: local models run in-process; API mode uses the selected provider profile.",
      "Tip: use specific questions about visible, observable facts. Accuracy depends on the selected model.",
    ].join("\n");
  }

  return [
    `Decision Router (${inputKind})`,
    "Choose exactly one matching route for each input item.",
    "Route name: output label. Route description: criteria the model uses to choose it.",
    "Batch inputs are evaluated item by item; items sent to each route keep their original order.",
    "An empty route output is blocked, so its downstream branch is skipped.",
    "decision_json contains the chosen route and probabilities.",
    "Backend: local models run in-process; API mode uses the selected provider profile.",
    "Tip: use clear, distinct descriptions, including when no route should match.",
  ].join("\n");
}

function ensureNotesWidget(node) {
  if (!Array.isArray(node.widgets)) return null;
  let widget = node.widgets.find((candidate) => candidate?.name === NOTES_WIDGET_NAME && candidate?.inputEl)
    ?? node.widgets.find((candidate) => candidate?.name === NOTES_WIDGET_NAME);
  if (!widget) {
    widget = ComfyWidgets.STRING(node, NOTES_WIDGET_NAME, ["STRING", { multiline: true }], app).widget;
  }
  if (widget?.inputEl) {
    widget.inputEl.readOnly = true;
    widget.inputEl.style.opacity = 0.78;
    widget.inputEl.style.fontSize = "12px";
    widget.inputEl.style.lineHeight = "1.35";
    widget.inputEl.rows = 10;
  }
  widget.options = { ...(widget.options || {}), serialize: false };
  widget.serializeValue = async () => undefined;
  widget.value = buildDecisionNotes(node);
  return widget;
}

function resizeNode(node) {
  requestAnimationFrame(() => {
    const size = node.computeSize();
    if (size[0] < node.size[0]) size[0] = node.size[0];
    if (size[1] < node.size[1]) size[1] = node.size[1];
    node.setSize(size);
    app.graph?.setDirtyCanvas?.(true, false);
  });
}

app.registerExtension({
  name: "ThinkingLLM.DecisionNotes",
  nodeCreated(node) {
    if (!DECISION_NODE_NAMES.has(node.comfyClass) || node._thinkingllmDecisionNotesHooked) return;
    node._thinkingllmDecisionNotesHooked = true;
    if (ensureNotesWidget(node)) resizeNode(node);
  },
});
