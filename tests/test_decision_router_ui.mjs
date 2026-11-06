import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const communitySource = await readFile(path.join(root, "web/js/decision_router.js"), "utf8");
const commercialSource = await readFile(path.join(root, "web_commercial/js/decision_router.js"), "utf8");
assert.equal(
  communitySource.replace("ThinkingLLM.DecisionRouter\"", "ThinkingLLM.DecisionRouter.Commercial\""),
  commercialSource,
  "community and commercial router behavior should stay in sync",
);

const extensionSource = communitySource
  .replace(/^import\s+\{\s*app\s*\}\s+from\s+["']\/scripts\/app\.js["'];\s*/m, "");
let extension;
const scheduled = [];
const app = { registerExtension(value) { extension = value; }, graph: null };
vm.runInNewContext(extensionSource, {
  app,
  setTimeout(callback) { scheduled.push(callback); },
  console: { error() {}, warn() {} },
});

const maxRoutes = 52;

function makeNodeType() {
  return { prototype: {} };
}

function makeWidgets() {
  const widgets = [];
  for (let index = 1; index <= maxRoutes; index += 1) {
    for (const field of ["name", "description"]) {
      widgets.push({
        name: `route_${String(index).padStart(2, "0")}_${field}`,
        value: index <= 2 && field === "name" ? `route_${index}` : "",
        callback(value) { this.value = value; },
      });
    }
  }
  return widgets;
}

function makeNode(routeType, outputs, graph = { links: {} }) {
  return {
    id: 44,
    graph,
    widgets: makeWidgets(),
    outputs: outputs.map((output, index) => typeof output === "string"
      ? { name: `slot_${index}`, type: output, links: [] }
      : output),
    addOutput(name, type) { this.outputs.push({ name, type, links: [] }); },
    removeOutput(index) { this.outputs.splice(index, 1); },
    computeSize() { return [320, this.outputs.length * 10]; },
    setDirtyCanvas() {},
    routeType,
  };
}

function widget(node, name) {
  return node.widgets.find((item) => item.name === name);
}

function flushScheduled() {
  while (scheduled.length) scheduled.splice(0).forEach((callback) => callback());
}

async function register(nodeName) {
  const nodeType = makeNodeType();
  await extension.beforeRegisterNodeDef(nodeType, { name: nodeName });
  return nodeType;
}

test("new Image and Text routers keep route_index before dynamic route sockets", async () => {
  for (const [nodeName, routeType] of [
    ["ThinkingLLM_DecisionRouter", "IMAGE"],
    ["ThinkingLLM_DecisionRouterText", "STRING"],
  ]) {
    const nodeType = await register(nodeName);
    const node = makeNode(routeType, ["STRING", "INT", ...Array(maxRoutes).fill(routeType)]);
    nodeType.prototype.onNodeCreated.call(node);
    flushScheduled();

    assert.deepEqual(node.outputs.map(({ name, type }) => [name, type]), [
      ["decision_json", "STRING"],
      ["route_index", "INT"],
      ["route_1", routeType],
      ["route_2", routeType],
    ]);

    widget(node, "route_03_name").callback.call(widget(node, "route_03_name"), "other");
    assert.equal(node.outputs[4].name, "other");
    assert.equal(node.outputs[4].type, routeType);
  }
});

test("legacy Image and Text router links shift one slot when route_index is introduced", async () => {
  for (const [nodeName, routeType] of [
    ["ThinkingLLM_DecisionRouter", "IMAGE"],
    ["ThinkingLLM_DecisionRouterText", "STRING"],
  ]) {
    const nodeType = await register(nodeName);
    const graph = {
      links: {
        501: { id: 501, origin_id: 44, origin_slot: 1 },
        502: { id: 502, origin_id: 44, origin_slot: 2 },
      },
    };
    const node = makeNode(routeType, [
      { name: "decision_json", type: "STRING", links: [] },
      { name: "cat", type: routeType, links: [501] },
      { name: "dog", type: routeType, links: [502] },
    ], graph);
    widget(node, "route_01_name").value = "cat";
    widget(node, "route_02_name").value = "dog";

    nodeType.prototype.onConfigure.call(node);
    flushScheduled();

    assert.deepEqual(node.outputs.map(({ name, type }) => [name, type]), [
      ["decision_json", "STRING"],
      ["route_index", "INT"],
      ["cat", routeType],
      ["dog", routeType],
    ]);
    assert.equal(graph.links[501].origin_slot, 2);
    assert.equal(graph.links[502].origin_slot, 3);
    assert.deepEqual(node.outputs[2].links, [501]);
    assert.deepEqual(node.outputs[3].links, [502]);
  }
});

test("new-schema workflows keep existing route link indexes", async () => {
  const nodeType = await register("ThinkingLLM_DecisionRouterText");
  const graph = { links: { 601: { id: 601, origin_id: 44, origin_slot: 3 } } };
  const node = makeNode("STRING", [
    { name: "decision_json", type: "STRING", links: [] },
    { name: "route_index", type: "INT", links: [] },
    { name: "cat", type: "STRING", links: [] },
    { name: "dog", type: "STRING", links: [601] },
  ], graph);

  nodeType.prototype.onConfigure.call(node);
  flushScheduled();

  assert.equal(node.outputs[1].name, "route_index");
  assert.equal(node.outputs[1].type, "INT");
  assert.equal(graph.links[601].origin_slot, 3);
  assert.equal(node.outputs[3].name, "route_2");
});
