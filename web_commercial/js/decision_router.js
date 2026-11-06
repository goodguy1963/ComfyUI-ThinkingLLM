import { app } from "/scripts/app.js";

const NODE_OUTPUT_TYPES = {
  ThinkingLLM_DecisionRouter: "IMAGE",
  ThinkingLLM_DecisionRouterText: "STRING",
};
const DECISION_FIELD_INPUT_KINDS = {
  ThinkingLLM_DecisionFieldsImage: "image",
  ThinkingLLM_DecisionFieldsText: "text",
};
const MAX_ROUTES = 52;
const ROUTE_INDEX_OUTPUT = "route_index";
const API_MODELS_URL = "/thinkingllm/decision/models";
const apiModelRequestIds = new WeakMap();
const pendingLinkSlots = new WeakMap();

function findWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function widgetValue(widget) {
  return String(widget?.value ?? "").trim();
}

async function refreshApiModels(node, inputKind) {
  const profileWidget = findWidget(node, "api_profile");
  const modelWidget = findWidget(node, "api_model");
  if (!profileWidget || !modelWidget || widgetValue(findWidget(node, "decision_backend")) !== "API") return;
  const profile = widgetValue(profileWidget);
  const requestId = (apiModelRequestIds.get(node) ?? 0) + 1;
  apiModelRequestIds.set(node, requestId);
  const query = new URLSearchParams({ profile, input_kind: inputKind });
  try {
    const response = await fetch(`${API_MODELS_URL}?${query.toString()}`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const payload = await response.json();
    if (apiModelRequestIds.get(node) !== requestId
      || widgetValue(findWidget(node, "decision_backend")) !== "API"
      || widgetValue(findWidget(node, "api_profile")) !== profile) return;
    const ids = (Array.isArray(payload.models) ? payload.models : [])
      .map((model) => String(model?.id ?? "")).filter(Boolean);
    if (!ids.length) throw new Error("No compatible decision models returned");
    modelWidget.type = "combo";
    modelWidget.serialize = true;
    modelWidget.options ??= {};
    modelWidget.options.values = ids;
    if (!ids.includes(widgetValue(modelWidget))) {
      modelWidget.value = ids[0];
      modelWidget.options.value = ids[0];
    }
  } catch (error) {
    if (apiModelRequestIds.get(node) !== requestId
      || widgetValue(findWidget(node, "decision_backend")) !== "API"
      || widgetValue(findWidget(node, "api_profile")) !== profile) return;
    console.warn("[ThinkingLLM Decision Router] Could not load API decision models.", error);
    modelWidget.type = "text";
  }
  node.size = node.computeSize?.() ?? node.size;
  node.setDirtyCanvas?.(true, true);
}

function synchronizeBackend(node, inputKind) {
  const apiMode = widgetValue(findWidget(node, "decision_backend")) === "API";
  for (const name of ["decision_model", "api_profile", "api_model", "keep_model_loaded", "include_reason"]) {
    const widget = findWidget(node, name);
    if (widget) widget.hidden = apiMode
      ? ["decision_model", "keep_model_loaded", "include_reason"].includes(name)
      : ["api_profile", "api_model"].includes(name);
  }
  if (apiMode) refreshApiModels(node, inputKind);
  node.size = node.computeSize?.() ?? node.size;
  node.setDirtyCanvas?.(true, true);
}

function widgetName(index, field) {
  return `route_${String(index + 1).padStart(2, "0")}_${field}`;
}

function routeValues(node) {
  return Array.from({ length: MAX_ROUTES }, (_, index) => ({
    name: String(node.widgets?.find((widget) => widget.name === widgetName(index, "name"))?.value ?? "").trim(),
    description: String(node.widgets?.find((widget) => widget.name === widgetName(index, "description"))?.value ?? "").trim(),
  }));
}

function linkId(link) {
  if (Array.isArray(link)) return link[0];
  return link && typeof link === "object" ? link.id : link;
}

function linkOrigin(link) {
  return Array.isArray(link) ? link[1] : link?.origin_id;
}

function setLinkOriginSlot(link, slot) {
  if (Array.isArray(link)) link[2] = slot;
  else if (link && typeof link === "object") link.origin_slot = slot;
}

function graphLinks(graph) {
  const links = graph?.links;
  if (links instanceof Map) return Array.from(links.values());
  if (Array.isArray(links)) return links.filter(Boolean);
  if (links && typeof links === "object") return Object.values(links).filter(Boolean);
  return [];
}

function repairLinks(node) {
  const remappedSlots = pendingLinkSlots.get(node);
  if (!remappedSlots) return;
  const graph = node.graph ?? app.graph;
  for (const link of graphLinks(graph)) {
    const id = linkId(link);
    if (String(linkOrigin(link)) !== String(node.id) || !remappedSlots.has(id)) continue;
    setLinkOriginSlot(link, remappedSlots.get(id));
    remappedSlots.delete(id);
  }
  if (!remappedSlots.size) pendingLinkSlots.delete(node);
}

function createOutput(node, name, type) {
  if (typeof node.addOutput === "function") {
    node.addOutput(name, type);
    return node.outputs.pop();
  }
  return { name, type, links: [] };
}

function synchronize(node, routeType) {
  if (!node || !Array.isArray(node.widgets) || !Array.isArray(node.outputs)) return;

  const oldOutputs = node.outputs.slice();
  const hasRouteIndex = oldOutputs[1]?.type === "INT";
  const oldRouteOutputs = oldOutputs.slice(hasRouteIndex ? 2 : 1);
  const routes = routeValues(node);
  let highestConfigured = -1;
  routes.forEach((route, index) => {
    if (route.name || route.description) highestConfigured = index;
  });

  const visibleRows = Math.min(MAX_ROUTES, Math.max(3, highestConfigured + 2));
  for (let index = 0; index < MAX_ROUTES; index += 1) {
    for (const field of ["name", "description"]) {
      const widget = node.widgets?.find((item) => item.name === widgetName(index, field));
      if (widget) widget.hidden = index >= visibleRows;
    }
  }

  const linkedThrough = oldRouteOutputs.reduce((highest, output, index) => {
    return output?.links?.length ? Math.max(highest, index) : highest;
  }, -1);
  const routeCount = Math.min(MAX_ROUTES, Math.max(0, highestConfigured + 1, linkedThrough + 1));
  const jsonOutput = oldOutputs[0] ?? createOutput(node, "decision_json", "STRING");
  const routeIndexOutput = hasRouteIndex
    ? oldOutputs[1]
    : createOutput(node, ROUTE_INDEX_OUTPUT, "INT");
  const nextOutputs = [jsonOutput, routeIndexOutput];
  const linkSlots = new Map();

  jsonOutput.name = "decision_json";
  jsonOutput.label = "decision_json";
  jsonOutput.type = "STRING";
  routeIndexOutput.name = ROUTE_INDEX_OUTPUT;
  routeIndexOutput.label = ROUTE_INDEX_OUTPUT;
  routeIndexOutput.type = "INT";

  oldRouteOutputs.forEach((output, slot) => {
    for (const link of output?.links ?? []) {
      const id = linkId(link);
      if (id !== undefined && id !== null) linkSlots.set(id, slot + 2);
    }
  });

  for (let slot = 0; slot < routeCount; slot += 1) {
    const route = routes[slot];
    const output = oldRouteOutputs[slot]
      ?? createOutput(node, route?.name || `route_${String(slot + 1).padStart(2, "0")}`, routeType);
    const name = route?.name || `inactive route_${String(slot + 1).padStart(2, "0")}`;
    output.name = name;
    output.label = name;
    output.type = routeType;
    nextOutputs.push(output);
  }

  node.outputs.splice(0, node.outputs.length, ...nextOutputs);
  if (linkSlots.size) pendingLinkSlots.set(node, linkSlots);
  repairLinks(node);

  if (typeof node.computeSize === "function") node.size = node.computeSize();
  node.setDirtyCanvas?.(true, true);
}

function safeSynchronize(node, routeType) {
  try {
    synchronize(node, routeType);
  } catch (error) {
    console.error("[ThinkingLLM Decision Router] Could not synchronize route sockets.", error);
  }
}

function scheduleSynchronize(node, routeType) {
  setTimeout(() => safeSynchronize(node, routeType), 0);
}

function afterLifecycle(node, routeType, result, callback) {
  if (result && typeof result.then === "function") {
    return result.then((value) => {
      callback(node, routeType);
      return value;
    });
  }
  callback(node, routeType);
  return result;
}

app.registerExtension({
  name: "ThinkingLLM.DecisionRouter.Commercial",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    const routeType = NODE_OUTPUT_TYPES[nodeData.name];
    const inputKind = DECISION_FIELD_INPUT_KINDS[nodeData.name]
      ?? (nodeData.name === "ThinkingLLM_DecisionRouter" ? "image" : routeType ? "text" : null);
    if (!inputKind) return;

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalCreated?.apply(this, arguments);
      const node = this;
      return afterLifecycle(node, routeType, result, (createdNode, outputType) => {
        for (const widget of createdNode.widgets ?? []) {
          if (["decision_backend", "api_profile"].includes(widget.name)) {
            const originalCallback = widget.callback;
            widget.callback = function (value, ...args) {
              const callbackResult = originalCallback?.call(this, value, ...args);
              synchronizeBackend(createdNode, inputKind);
              return callbackResult;
            };
            continue;
          }
          if (!routeType || !widget.name?.startsWith("route_")) continue;
          const originalCallback = widget.callback;
          widget.callback = function (value, ...args) {
            const callbackResult = originalCallback?.call(this, value, ...args);
            return afterLifecycle(createdNode, outputType, callbackResult, safeSynchronize);
          };
        }
        if (outputType) scheduleSynchronize(createdNode, outputType);
        synchronizeBackend(createdNode, inputKind);
      });
    };

    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = originalConfigure?.apply(this, arguments);
      return afterLifecycle(this, routeType, result, (configuredNode, outputType) => {
        if (outputType) scheduleSynchronize(configuredNode, outputType);
        synchronizeBackend(configuredNode, inputKind);
      });
    };
  },
});
