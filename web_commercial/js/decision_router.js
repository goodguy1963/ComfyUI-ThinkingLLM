import { app } from "/scripts/app.js";

const NODE_OUTPUT_TYPES = {
  ThinkingLLM_DecisionRouter: "IMAGE",
  ThinkingLLM_DecisionRouterText: "STRING",
};
const MAX_ROUTES = 52;
const API_MODELS_URL = "/thinkingllm/decision/models";
const apiModelRequestIds = new WeakMap();

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

function synchronize(node, routeType) {
  if (!node || !Array.isArray(node.widgets) || !Array.isArray(node.outputs)) return;

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

  const linkedThrough = (node.outputs ?? []).reduce((highest, output, index) => {
    return output?.links?.length ? Math.max(highest, index) : highest;
  }, 0);
  const configuredThrough = highestConfigured + 1;
  const wantedCount = Math.max(1, configuredThrough + 1, linkedThrough + 1);

  while ((node.outputs?.length ?? 0) < wantedCount) {
    const index = node.outputs.length;
    const slot = index - 1;
    if (typeof node.addOutput !== "function") break;
    node.addOutput(`route_${String(slot + 1).padStart(2, "0")}`, routeType);
  }
  while ((node.outputs?.length ?? 0) > wantedCount) {
    const lastIndex = node.outputs.length - 1;
    if (node.outputs[lastIndex]?.links?.length || typeof node.removeOutput !== "function") break;
    node.removeOutput(lastIndex);
  }

  if (node.outputs?.[0]) {
    node.outputs[0].name = "decision_json";
    node.outputs[0].label = "decision_json";
    node.outputs[0].type = "STRING";
  }
  for (let index = 1; index < (node.outputs?.length ?? 0); index += 1) {
    const route = routes[index - 1];
    const name = route?.name || `inactive route_${String(index).padStart(2, "0")}`;
    node.outputs[index].name = name;
    node.outputs[index].label = name;
    node.outputs[index].type = routeType;
  }

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
    if (!routeType) return;
    const inputKind = nodeData.name === "ThinkingLLM_DecisionRouter" ? "image" : "text";

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
          if (!widget.name?.startsWith("route_")) continue;
          const originalCallback = widget.callback;
          widget.callback = function (value, ...args) {
            const callbackResult = originalCallback?.call(this, value, ...args);
            return afterLifecycle(createdNode, outputType, callbackResult, safeSynchronize);
          };
        }
        scheduleSynchronize(createdNode, outputType);
        synchronizeBackend(createdNode, inputKind);
      });
    };

    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = originalConfigure?.apply(this, arguments);
      return afterLifecycle(this, routeType, result, (configuredNode, outputType) => {
        scheduleSynchronize(configuredNode, outputType);
        synchronizeBackend(configuredNode, inputKind);
      });
    };
  },
});
