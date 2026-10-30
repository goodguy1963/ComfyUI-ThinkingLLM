import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
let extension;
const app = { registerExtension(value) { extension = value; }, graph: null };
const comfyWidgets = {
  STRING(node, name) {
    const widget = {
      name,
      value: "",
      options: {},
      inputEl: { readOnly: false, style: {}, rows: 0 },
    };
    node.widgets.push(widget);
    return { widget };
  },
};
const communitySource = await readFile(path.join(root, "web/js/decision_notes.js"), "utf8");
const commercialSource = await readFile(path.join(root, "web_commercial/js/decision_notes.js"), "utf8");
assert.equal(communitySource, commercialSource, "community and commercial builds should show the same Decision notes");
const source = communitySource
  .replace(/^import\s+\{\s*app\s*\}\s+from\s+["']\/scripts\/app\.js["'];\s*/m, "")
  .replace(/^import\s+\{\s*ComfyWidgets\s*\}\s+from\s+["']\/scripts\/widgets\.js["'];\s*/m, "");

vm.runInNewContext(source, {
  app,
  ComfyWidgets: comfyWidgets,
  requestAnimationFrame(callback) { callback(); },
  console,
});

function createNode(comfyClass) {
  return {
    comfyClass,
    widgets: [],
    size: [360, 200],
    computeSize() { return [360, 420]; },
    setSize(size) { this.size = size; },
  };
}

function notesFor(comfyClass) {
  const node = createNode(comfyClass);
  extension.nodeCreated(node);
  const widget = node.widgets.find((candidate) => candidate.name === "recommended_settings");
  assert.ok(widget, `${comfyClass} should receive the shared notes widget`);
  assert.equal(widget.inputEl.readOnly, true);
  assert.equal(widget.options.serialize, false);
  return widget.value;
}

test("Decision Router notes explain route names, descriptions, batches, and blocked branches", () => {
  const imageNotes = notesFor("ThinkingLLM_DecisionRouter");
  const textNotes = notesFor("ThinkingLLM_DecisionRouterText");
  assert.match(imageNotes, /Decision Router \(image\)/);
  assert.match(textNotes, /Decision Router \(text\)/);
  assert.match(imageNotes, /exactly one matching route/);
  assert.match(imageNotes, /Route description: criteria/);
  assert.match(imageNotes, /evaluated item by item/);
  assert.match(imageNotes, /downstream branch is skipped/);
});

test("Decision Fields notes define Boolean, threshold, Integer, and summary outputs", () => {
  const notes = notesFor("ThinkingLLM_DecisionFieldsImage");
  const textNotes = notesFor("ThinkingLLM_DecisionFieldsText");
  assert.match(notes, /Decision Fields \(image\)/);
  assert.match(textNotes, /Decision Fields \(text\)/);
  assert.match(notes, /BOOLEAN: ask a yes\/no question/);
  assert.match(notes, /Threshold \(0-1\)/);
  assert.match(notes, /INT: choose a whole number/);
  assert.match(notes, /Minimum through Maximum, inclusive/);
  assert.match(notes, /decision_json: detailed machine-readable values/);
  assert.match(notes, /decision: readable multiline summary/);
});
