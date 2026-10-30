import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const extensionSource = (await readFile(path.join(root, "web/js/decision_fields.js"), "utf8"))
  .replace(/^import\s+\{\s*app\s*\}\s+from\s+["']\/scripts\/app\.js["'];\s*/m, "");
let extension;
const scheduled = [];
const app = { registerExtension(value) { extension = value; }, graph: null };
vm.runInNewContext(extensionSource, {
  app,
  setTimeout(callback) { scheduled.push(callback); },
  console: { error() {}, warn() {} },
});

const maxFields = 8;
const anyOutputTypes = ["STRING", "STRING", ...Array(maxFields * 2).fill("*")];
const legacyBooleanFirst = ["STRING", ...Array(maxFields).fill("BOOLEAN"), ...Array(maxFields).fill("INT")];
const previousIntegerFirst = ["STRING", ...Array(maxFields).fill("INT"), "BOOLEAN", "BOOLEAN"];

function makeNodeType() {
  return { prototype: {} };
}

function makeWidgets(outputSlots = "[]") {
  const widgets = [];
  for (const group of ["boolean", "integer"]) {
    for (let index = 1; index <= maxFields; index += 1) {
      const parts = group === "boolean" ? ["name", "question", "threshold"] : ["name", "question", "minimum", "maximum"];
      for (const part of parts) {
        widgets.push({
          name: `${group}_${String(index).padStart(2, "0")}_${part}`,
          value: "",
          callback(value) { this.value = value; },
        });
      }
    }
  }
  widgets.push({ name: "decision_output_slots", value: outputSlots, hidden: false, serialize: true });
  return widgets;
}

function makeNode(outputs, graph = { links: {} }, outputSlots = "[]") {
  return {
    id: 44,
    graph,
    properties: {},
    size: [320, 200],
    computeSizeCalls: 0,
    widgets: makeWidgets(outputSlots),
    outputs: outputs.map((type, index) => ({ name: `slot_${index}`, type, links: [] })),
    addOutput(name, type) { this.outputs.push({ name, type, links: [] }); },
    removeOutput(index) { this.outputs.splice(index, 1); },
    computeSize() {
      this.computeSizeCalls += 1;
      const longestOutput = Math.max(0, ...this.outputs.map((output) => String(output.name ?? "").length));
      return [Math.max(320, longestOutput * 12), this.outputs.length * 10];
    },
    setDirtyCanvas() {},
  };
}

function findWidget(node, name) {
  return node.widgets.find((widget) => widget.name === name);
}

function setWidget(node, name, value) {
  const widget = findWidget(node, name);
  if (widget.callback) widget.callback.call(widget, value);
  else widget.value = value;
}

function fieldKey(group, index) {
  return `${group}_${String(index).padStart(2, "0")}`;
}

test("Boolean and Integer sockets appear only for configured fields and links follow reordered slots", async () => {
  const nodeType = makeNodeType();
  await extension.beforeRegisterNodeDef(nodeType, { name: "ThinkingLLM_DecisionFieldsImage" });
  const node = makeNode(anyOutputTypes);
  nodeType.prototype.onNodeCreated.call(node);

  assert.equal(node.outputs.length, 2);
  assert.deepEqual(node.outputs.slice(0, 2).map((output) => [output.name, output.type]), [
    ["decision_json", "STRING"],
    ["decision", "STRING"],
  ]);
  assert.equal(findWidget(node, "decision_output_slots").hidden, true);
  assert.equal(findWidget(node, "boolean_01_name").hidden, false);
  assert.equal(findWidget(node, "boolean_02_name").hidden, true);

  setWidget(node, "boolean_01_name", "fox_visible");
  setWidget(node, "boolean_01_question", "Is a fox visible?");
  const afterFirstFieldLayout = node.computeSizeCalls;
  const fixedWidth = node.size[0];
  setWidget(node, "boolean_01_name", "fox_visible_with_a_longer_name");
  setWidget(node, "boolean_01_question", "Is a fox visible in this image, including small or distant foxes?");
  assert.equal(node.computeSizeCalls, afterFirstFieldLayout, "editing existing field text must not trigger size recalculation");
  assert.equal(node.size[0], fixedWidth, "longer text must not make the node wider");
  setWidget(node, "boolean_01_name", "fox_visible");
  setWidget(node, "boolean_01_question", "Is a fox visible?");
  assert.deepEqual(node.outputs.slice(2).map((output) => [output.name, output.type]), [["fox_visible", "BOOLEAN"]]);
  setWidget(node, "boolean_02_name", "dog_visible");
  setWidget(node, "boolean_02_question", "Is a dog visible?");
  assert.equal(node.outputs.length, 4);
  assert.equal(node.outputs[3].type, "BOOLEAN");
  assert.equal(findWidget(node, "boolean_03_name").hidden, false);
  assert.equal(findWidget(node, "boolean_04_name").hidden, true);

  setWidget(node, "integer_01_name", "people_count");
  setWidget(node, "integer_01_question", "How many people are visible?");
  assert.deepEqual(node.outputs.slice(2).map((output) => [output.name, output.type]), [
    ["fox_visible", "BOOLEAN"],
    ["dog_visible", "BOOLEAN"],
    ["people_count", "INT"],
  ]);
  assert.equal(findWidget(node, "decision_output_slots").value, '["boolean_01","boolean_02","integer_01"]');

  node.outputs[4].links = [501];
  node.graph.links[501] = { id: 501, origin_id: node.id, origin_slot: 4 };
  setWidget(node, "boolean_03_name", "cat_visible");
  setWidget(node, "boolean_03_question", "Is a cat visible?");
  assert.equal(node.graph.links[501].origin_slot, 5, "adding a Boolean before an Integer preserves the Integer connection");
  assert.equal(node.outputs[5].name, "people_count");

  setWidget(node, "integer_01_name", "");
  setWidget(node, "integer_01_question", "");
  assert.equal(node.outputs.length, 6, "a connected but unconfigured Integer socket stays in the graph");
  assert.match(node.outputs[5].name, /unused/);
  node.outputs[5].links = [];
  delete node.graph.links[501];
  nodeType.prototype.onConnectionsChange.call(node);
  scheduled.splice(0).forEach((callback) => callback());
  assert.equal(node.outputs.length, 5);

  for (let index = 4; index <= maxFields; index += 1) {
    const suffix = String(index).padStart(2, "0");
    setWidget(node, `boolean_${suffix}_name`, `condition_${index}`);
    setWidget(node, `boolean_${suffix}_question`, `Is condition ${index} true?`);
  }
  for (let index = 1; index <= maxFields; index += 1) {
    const suffix = String(index).padStart(2, "0");
    setWidget(node, `integer_${suffix}_name`, `number_${index}`);
    setWidget(node, `integer_${suffix}_question`, `What is number ${index}?`);
  }
  assert.equal(node.outputs.length, 2 + maxFields * 2);
  assert.equal(node.outputs.filter((output) => output.type === "BOOLEAN").length, maxFields);
  assert.equal(node.outputs.filter((output) => output.type === "INT").length, maxFields);
});

test("legacy Boolean-first and recent Integer-first workflows migrate links by their original socket types", async () => {
  const nodeType = makeNodeType();
  await extension.beforeRegisterNodeDef(nodeType, { name: "ThinkingLLM_DecisionFieldsText" });

  for (const [types, boolSlot, intSlot] of [
    [legacyBooleanFirst, 1, 1 + maxFields],
    [previousIntegerFirst, 1 + maxFields, 1],
  ]) {
    const graph = {
      links: {
        701: { id: 701, origin_id: 44, origin_slot: boolSlot },
        702: { id: 702, origin_id: 44, origin_slot: intSlot },
      },
    };
    const node = makeNode(types, graph);
    node.outputs[boolSlot].links = [701];
    node.outputs[intSlot].links = [702];
    findWidget(node, "boolean_01_name").value = "fox_visible";
    findWidget(node, "boolean_01_question").value = "Is a fox visible?";
    findWidget(node, "integer_01_name").value = "people_count";
    findWidget(node, "integer_01_question").value = "How many people are visible?";

    nodeType.prototype.onConfigure.call(node);

    assert.deepEqual(node.outputs.slice(2).map((output) => [output.name, output.type]), [
      ["fox_visible", "BOOLEAN"],
      ["people_count", "INT"],
    ]);
    assert.equal(graph.links[701].origin_slot, 2);
    assert.equal(graph.links[702].origin_slot, 3);
  }
});

test("saved dynamic field slots restore their names, native types, and link indexes", async () => {
  const nodeType = makeNodeType();
  await extension.beforeRegisterNodeDef(nodeType, { name: "ThinkingLLM_DecisionFieldsImage" });
  const original = makeNode(anyOutputTypes);
  nodeType.prototype.onNodeCreated.call(original);
  setWidget(original, "boolean_01_name", "fox_visible");
  setWidget(original, "boolean_01_question", "Is a fox visible?");
  setWidget(original, "integer_01_name", "people_count");
  setWidget(original, "integer_01_question", "How many people are visible?");
  original.outputs[3].links = [703];

  const savedKeys = findWidget(original, "decision_output_slots").value;
  const savedOutputs = original.outputs.map((output) => ({
    name: output.name,
    type: output.type,
    links: [...(output.links ?? [])],
  }));
  const graph = { links: { 703: { id: 703, origin_id: 44, origin_slot: 3 } } };
  const restored = makeNode(savedOutputs.map((output) => output.type), graph, savedKeys);
  restored.outputs = savedOutputs;
  for (const [name, value] of [
    ["boolean_01_name", "fox_visible"],
    ["boolean_01_question", "Is a fox visible?"],
    ["integer_01_name", "people_count"],
    ["integer_01_question", "How many people are visible?"],
  ]) {
    findWidget(restored, name).value = value;
  }

  nodeType.prototype.onConfigure.call(restored);

  assert.deepEqual(restored.outputs.slice(2).map((output) => [output.name, output.type]), [
    ["fox_visible", "BOOLEAN"],
    ["people_count", "INT"],
  ]);
  assert.equal(graph.links[703].origin_slot, 3);
  assert.equal(findWidget(restored, "decision_output_slots").hidden, true);
});

test("adding the readable output migrates links from the previous dynamic slot schema", async () => {
  const nodeType = makeNodeType();
  await extension.beforeRegisterNodeDef(nodeType, { name: "ThinkingLLM_DecisionFieldsImage" });
  const graph = {
    links: {
      801: { id: 801, origin_id: 44, origin_slot: 1 },
      802: { id: 802, origin_id: 44, origin_slot: 2 },
    },
  };
  const savedOutputs = [
    { name: "decision_json", type: "STRING", links: [] },
    { name: "fox_visible", type: "BOOLEAN", links: [801] },
    { name: "people_count", type: "INT", links: [802] },
  ];
  const restored = makeNode(savedOutputs.map((output) => output.type), graph, '["boolean_01","integer_01"]');
  restored.outputs = savedOutputs;
  for (const [name, value] of [
    ["boolean_01_name", "fox_visible"],
    ["boolean_01_question", "Is a fox visible?"],
    ["integer_01_name", "people_count"],
    ["integer_01_question", "How many people are visible?"],
  ]) {
    findWidget(restored, name).value = value;
  }

  nodeType.prototype.onConfigure.call(restored);

  assert.deepEqual(restored.outputs.map((output) => [output.name, output.type]), [
    ["decision_json", "STRING"],
    ["decision", "STRING"],
    ["fox_visible", "BOOLEAN"],
    ["people_count", "INT"],
  ]);
  assert.equal(graph.links[801].origin_slot, 2);
  assert.equal(graph.links[802].origin_slot, 3);
  assert.equal(findWidget(restored, "decision_output_slots").value, '["boolean_01","integer_01"]');
});
