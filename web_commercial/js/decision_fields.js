import { app } from "/scripts/app.js";

const FIELD_NODES = {
  ThinkingLLM_DecisionFieldsImage: true,
  ThinkingLLM_DecisionFieldsText: true,
};
const MAX_FIELDS_PER_TYPE = 8;
const OUTPUT_SCHEMA_WIDGET = "decision_output_slots";
const pendingLinkSlots = new WeakMap();
const layoutSignatures = new WeakMap();

function findWidget(node, name) {
  return node.widgets?.find((widget) => widget.name === name);
}

function widgetValue(node, name) {
  return String(findWidget(node, name)?.value ?? "").trim();
}

function fieldWidgetName(type, index, part) {
  return `${type}_${String(index + 1).padStart(2, "0")}_${part}`;
}

function fieldKey(type, index) {
  return `${type}_${String(index + 1).padStart(2, "0")}`;
}

function fixedOutputCount(node) {
  const summaryOutput = node.outputs?.[1];
  return summaryOutput?.name === "decision" && String(summaryOutput.type).toUpperCase() === "STRING" ? 2 : 1;
}

function parseFieldKey(value) {
  const match = /^(boolean|integer)_(\d{2})$/.exec(String(value ?? ""));
  if (!match) return null;
  const index = Number(match[2]) - 1;
  if (!Number.isInteger(index) || index < 0 || index >= MAX_FIELDS_PER_TYPE) return null;
  return { key: match[0], group: match[1], index, type: match[1] === "boolean" ? "BOOLEAN" : "INT" };
}

function parseOutputSchema(node) {
  const widget = findWidget(node, OUTPUT_SCHEMA_WIDGET);
  if (widget) {
    widget.hidden = true;
    widget.serialize = true;
  }
  try {
    const value = JSON.parse(String(widget?.value ?? "[]"));
    if (!Array.isArray(value) || value.some((key) => !parseFieldKey(key)) || new Set(value).size !== value.length) {
      return null;
    }
    if (value.length !== Math.max(0, (node.outputs?.length ?? 1) - fixedOutputCount(node))) return null;
    return value;
  } catch {
    return null;
  }
}

function legacyFieldKeys(node) {
  const outputs = node.outputs ?? [];
  const prefixCount = fixedOutputCount(node);
  const keys = new Map();
  const outputType = (index) => String(outputs[index]?.type ?? "").toUpperCase();
  const keyByName = new Map();
  for (const group of ["boolean", "integer"]) {
    for (let index = 0; index < MAX_FIELDS_PER_TYPE; index += 1) {
      const name = widgetValue(node, fieldWidgetName(group, index, "name")).toLocaleLowerCase();
      if (name) keyByName.set(name, fieldKey(group, index));
    }
  }
  for (let slot = prefixCount; slot < outputs.length; slot += 1) {
    const name = String(outputs[slot]?.name ?? "").trim();
    const placeholder = /^(boolean|integer)_\d{2}(?:\s+\(unused\))?$/.exec(name);
    const namedKey = placeholder
      ? `${placeholder[1]}_${name.slice(placeholder[1].length + 1, placeholder[1].length + 3)}`
      : keyByName.get(name.toLocaleLowerCase());
    if (parseFieldKey(namedKey)) keys.set(slot, namedKey);
  }
  const oldBooleanFirst = outputs.length >= prefixCount + MAX_FIELDS_PER_TYPE * 2
    && outputType(prefixCount) === "BOOLEAN"
    && outputType(prefixCount + MAX_FIELDS_PER_TYPE) === "INT";

  if (oldBooleanFirst) {
    for (let index = 0; index < MAX_FIELDS_PER_TYPE; index += 1) {
      if (!keys.has(index + prefixCount)) keys.set(index + prefixCount, fieldKey("boolean", index));
      if (!keys.has(index + prefixCount + MAX_FIELDS_PER_TYPE)) {
        keys.set(index + prefixCount + MAX_FIELDS_PER_TYPE, fieldKey("integer", index));
      }
    }
  } else if (outputType(prefixCount) === "INT") {
    for (let index = 0; index < MAX_FIELDS_PER_TYPE; index += 1) {
      if (outputType(index + prefixCount) === "INT" && !keys.has(index + prefixCount)) {
        keys.set(index + prefixCount, fieldKey("integer", index));
      }
    }
    for (let slot = prefixCount + MAX_FIELDS_PER_TYPE; slot < outputs.length; slot += 1) {
      const type = outputType(slot);
      if (type === "BOOLEAN" && !keys.has(slot)) keys.set(slot, fieldKey("boolean", slot - prefixCount - MAX_FIELDS_PER_TYPE));
      else if (type === "INT" && !keys.has(slot)) keys.set(slot, fieldKey("integer", slot - prefixCount));
    }
  }
  return keys;
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

function activeFields(node) {
  const fields = [];
  for (const group of ["boolean", "integer"]) {
    for (let index = 0; index < MAX_FIELDS_PER_TYPE; index += 1) {
      const name = widgetValue(node, fieldWidgetName(group, index, "name"));
      const question = widgetValue(node, fieldWidgetName(group, index, "question"));
      if (name || question) {
        fields.push({ key: fieldKey(group, index), group, index, type: group === "boolean" ? "BOOLEAN" : "INT", name });
      }
    }
  }
  return fields;
}

function sortFieldKeys(keys) {
  return keys.sort((left, right) => {
    const a = parseFieldKey(left);
    const b = parseFieldKey(right);
    if (a.group !== b.group) return a.group === "boolean" ? -1 : 1;
    return a.index - b.index;
  });
}

function synchronize(node) {
  if (!Array.isArray(node.widgets) || !Array.isArray(node.outputs)) return;

  const outputSchema = parseOutputSchema(node);
  const legacyKeys = outputSchema ? new Map() : legacyFieldKeys(node);
  const oldOutputs = node.outputs.slice();
  const oldPrefixCount = fixedOutputCount(node);
  const oldFieldOutputs = new Map();
  for (let slot = oldPrefixCount; slot < oldOutputs.length; slot += 1) {
    const output = oldOutputs[slot];
    const key = outputSchema?.[slot - oldPrefixCount]
      ?? output?._thinkingLlmFieldKey
      ?? legacyKeys.get(slot);
    if (!parseFieldKey(key)) continue;
    const existing = oldFieldOutputs.get(key);
    if (!existing || (!existing.links?.length && output?.links?.length)) oldFieldOutputs.set(key, output);
  }

  const jsonOutput = oldOutputs[0] ?? createOutput(node, "decision_json", "STRING");
  jsonOutput.name = "decision_json";
  jsonOutput.label = "decision_json";
  jsonOutput.type = "STRING";
  const decisionOutput = oldPrefixCount === 2
    ? oldOutputs[1]
    : createOutput(node, "decision", "STRING");
  decisionOutput.name = "decision";
  decisionOutput.label = "decision";
  decisionOutput.type = "STRING";

  const configuredFields = activeFields(node);
  const configuredByKey = new Map(configuredFields.map((field) => [field.key, field]));
  const desiredKeys = new Set(configuredByKey.keys());
  for (const [key, output] of oldFieldOutputs) {
    if (output?.links?.length) desiredKeys.add(key);
  }
  const orderedKeys = sortFieldKeys(Array.from(desiredKeys));
  const nextOutputs = [jsonOutput, decisionOutput];
  const linkSlots = new Map();

  for (const key of orderedKeys) {
    const field = configuredByKey.get(key);
    const parsed = parseFieldKey(key);
    const type = field?.type ?? parsed.type;
    const fallback = `${key} (unused)`;
    const output = oldFieldOutputs.get(key) ?? createOutput(node, field?.name || fallback, type);
    output._thinkingLlmFieldKey = key;
    output.type = type;
    output.name = field?.name || fallback;
    output.label = output.name;
    nextOutputs.push(output);
    for (const link of output.links ?? []) {
      linkSlots.set(linkId(link), nextOutputs.length - 1);
    }
  }

  node.outputs.splice(0, node.outputs.length, ...nextOutputs);
  const schemaWidget = findWidget(node, OUTPUT_SCHEMA_WIDGET);
  if (schemaWidget) schemaWidget.value = JSON.stringify(orderedKeys);
  if (linkSlots.size) pendingLinkSlots.set(node, linkSlots);
  repairLinks(node);

  const visibleRowsByGroup = {};
  for (const group of ["boolean", "integer"]) {
    let highestConfigured = -1;
    for (let index = 0; index < MAX_FIELDS_PER_TYPE; index += 1) {
      const name = widgetValue(node, fieldWidgetName(group, index, "name"));
      const question = widgetValue(node, fieldWidgetName(group, index, "question"));
      if (name || question) highestConfigured = index;
    }
    const visibleRows = Math.min(MAX_FIELDS_PER_TYPE, Math.max(1, highestConfigured + 2));
    visibleRowsByGroup[group] = visibleRows;
    const parts = group === "boolean"
      ? ["name", "question", "threshold"]
      : ["name", "question", "minimum", "maximum"];
    for (let index = 0; index < MAX_FIELDS_PER_TYPE; index += 1) {
      for (const part of parts) {
        const widget = findWidget(node, fieldWidgetName(group, index, part));
        if (widget) widget.hidden = index >= visibleRows;
      }
    }
  }

  const layoutSignature = JSON.stringify({ fields: orderedKeys, visibleRowsByGroup });
  if (layoutSignatures.get(node) !== layoutSignature) {
    if (typeof node.computeSize === "function") {
      const computedSize = node.computeSize();
      const currentWidth = Number(node.size?.[0]);
      const width = Number.isFinite(currentWidth) && currentWidth > 0
        ? currentWidth
        : Number(computedSize?.[0]) || 0;
      node.size = [width, computedSize?.[1] ?? node.size?.[1]];
    }
    layoutSignatures.set(node, layoutSignature);
  }
  node.setDirtyCanvas?.(true, true);
}

function safeSynchronize(node) {
  try {
    synchronize(node);
  } catch (error) {
    console.error("[ThinkingLLM Decision Fields] Could not synchronize dynamic field sockets.", error);
  }
}

function scheduleSynchronize(node) {
  setTimeout(() => safeSynchronize(node), 0);
}

function afterLifecycle(node, result) {
  if (result && typeof result.then === "function") {
    return result.then((value) => {
      safeSynchronize(node);
      return value;
    });
  }
  safeSynchronize(node);
  return result;
}

function attachFieldWidgetCallbacks(node) {
  for (const widget of node.widgets ?? []) {
    if (!/^(boolean|integer)_\d{2}_/.test(widget.name ?? "") || widget._thinkingLlmFieldCallback) continue;
    const originalCallback = widget.callback;
    widget.callback = function (value, ...args) {
      const result = originalCallback?.call(this, value, ...args);
      safeSynchronize(node);
      return result;
    };
    widget._thinkingLlmFieldCallback = true;
  }
}

app.registerExtension({
  name: "ThinkingLLM.DecisionFields",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (!FIELD_NODES[nodeData.name]) return;

    const originalCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const result = originalCreated?.apply(this, arguments);
      const node = this;
      return afterLifecycle(node, result && typeof result.then === "function"
        ? result.then((value) => {
          attachFieldWidgetCallbacks(node);
          return value;
        })
        : (attachFieldWidgetCallbacks(node), result));
    };

    const originalConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const result = originalConfigure?.apply(this, arguments);
      const node = this;
      const lifecycleResult = afterLifecycle(node, result && typeof result.then === "function"
        ? result.then((value) => {
          attachFieldWidgetCallbacks(node);
          return value;
        })
        : (attachFieldWidgetCallbacks(node), result));
      if (lifecycleResult && typeof lifecycleResult.then === "function") {
        return lifecycleResult.then((value) => {
          scheduleSynchronize(node);
          return value;
        });
      }
      scheduleSynchronize(node);
      return lifecycleResult;
    };

    const originalConnectionsChange = nodeType.prototype.onConnectionsChange;
    nodeType.prototype.onConnectionsChange = function () {
      const result = originalConnectionsChange?.apply(this, arguments);
      scheduleSynchronize(this);
      return result;
    };
  },
});
