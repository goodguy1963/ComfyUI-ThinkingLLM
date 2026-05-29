import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

const COLOR_THEMES = {
    QwenVL: { nodeColor: "#28403f", nodeBgColor: "#374539", width: 340 },
    QwenVLGGUF: { nodeColor: "#474539", nodeBgColor: "#2c4045", width: 340 },
    Tools: { nodeColor: "#28403f", nodeBgColor: "#233238", width: 300 },
    Enhancer: { nodeColor: "#374445", nodeBgColor: "#474539", width: 340 },
};

const NODE_COLORS = {
    "AILab_QwenVL": "QwenVL",
    "AILab_QwenVL_Advanced": "QwenVL",
    "AILab_QwenVL_PromptEnhancer": "Enhancer",
    "AILab_QwenVL_GGUF": "QwenVLGGUF",
    "AILab_QwenVL_GGUF_Advanced": "QwenVLGGUF",
    "AILab_QwenVL_GGUF_PromptEnhancer": "Enhancer",
    "AILab_QwenVL_PromptLibrary": "Tools",
};

const THINKINGLLM_NODE_NAMES = new Set([
    "AILab_QwenVL",
    "AILab_QwenVL_Advanced",
    "AILab_QwenVL_PromptEnhancer",
    "AILab_QwenVL_GGUF",
    "AILab_QwenVL_GGUF_Advanced",
    "AILab_QwenVL_GGUF_PromptEnhancer",
    "ThinkingLLM_QwenVL",
    "ThinkingLLM_QwenVL_Advanced",
    "ThinkingLLM_QwenVL_PromptEnhancer",
    "ThinkingLLM_QwenVL_GGUF",
    "ThinkingLLM_QwenVL_GGUF_Advanced",
    "ThinkingLLM_QwenVL_GGUF_PromptEnhancer",
]);

const GGUF_ADVANCED_INTERNAL_WIDGETS = new Set([
    "legacy_seed_mode",
    "legacy_unload_after_run",
    "n_ubatch",
    "n_threads",
    "n_threads_batch",
    "flash_attn",
    "offload_kqv",
    "ctx_checkpoints",
]);

const RECOMMENDATIONS_URL = new URL("../model_recommendations.json", import.meta.url);
const RECOMMENDATION_WIDGET_NAME = "recommended_settings";
const RECOMMENDATION_PLACEHOLDER = "Select a model to see provider/community recommended settings. This note never changes your saved widget values.";
const RECOMMENDATION_FIELDS = ["temperature", "top_p", "top_k", "max_tokens", "ctx_min", "n_batch", "image_max_tokens"];

let recommendationsPromise = null;

function loadRecommendations() {
    if (!recommendationsPromise) {
        recommendationsPromise = fetch(RECOMMENDATIONS_URL)
            .then((response) => {
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                return response.json();
            })
            .then((payload) => Array.isArray(payload?.rules) ? payload.rules : [])
            .catch((error) => {
                console.warn("[ThinkingLLM] Failed to load model recommendations", error);
                return [];
            });
    }
    return recommendationsPromise;
}

function setNodeColors(node, theme) {
    if (!theme) { return; }
    if (theme.nodeColor) {
        node.color = theme.nodeColor;
    }
    if (theme.nodeBgColor) {
        node.bgcolor = theme.nodeBgColor;
    }
    if (theme.width) {
        node.size = node.size || [140, 80];
        node.size[0] = theme.width;
    }
}

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function hideStableWidget(node, widget) {
    if (!widget || widget._thinkingllmHidden) { return; }
    widget._thinkingllmHidden = true;
    widget._thinkingllmOriginalType = widget.type;
    widget._thinkingllmOriginalComputeSize = widget.computeSize;
    widget.hidden = true;
    widget.type = `thinkingllm_hidden_${widget.type || "widget"}`;
    widget.computeSize = () => [0, -4];
}

function hideGgufAdvancedInternals(node) {
    if (!["AILab_QwenVL_GGUF_Advanced", "ThinkingLLM_QwenVL_GGUF_Advanced"].includes(node.comfyClass)) {
        return;
    }
    for (const widgetName of GGUF_ADVANCED_INTERNAL_WIDGETS) {
        hideStableWidget(node, findWidget(node, widgetName));
    }
    if (node.widgets?.length) {
        requestAnimationFrame(() => node.setSize([node.size[0], node.computeSize()[1]]));
    }
}

function clearHfTokenAfterExecution(node) {
    if (!THINKINGLLM_NODE_NAMES.has(node.comfyClass) || node._thinkingllmClearsToken) {
        return;
    }
    node._thinkingllmClearsToken = true;
    const originalOnExecuted = node.onExecuted;
    node.onExecuted = function (...args) {
        const result = originalOnExecuted?.apply(this, args);
        const tokenWidget = findWidget(this, "hf_token");
        if (tokenWidget?.value) {
            tokenWidget.value = "";
            app.graph?.setDirtyCanvas?.(true, true);
        }
        return result;
    };
}

function recommendationScope(node) {
    if ([
        "AILab_QwenVL_GGUF",
        "AILab_QwenVL_GGUF_Advanced",
        "ThinkingLLM_QwenVL_GGUF",
        "ThinkingLLM_QwenVL_GGUF_Advanced",
        "AILab_QwenVL_GGUF_PromptEnhancer",
        "ThinkingLLM_QwenVL_GGUF_PromptEnhancer",
    ].includes(node.comfyClass)) {
        return "gguf";
    }
    return "hf";
}

function matchRecommendationRule(modelName, rules) {
    const lowered = String(modelName || "").toLowerCase();
    if (!lowered) {
        return null;
    }
    const ordered = [...(rules || [])].sort((left, right) => (right.priority || 0) - (left.priority || 0));
    for (const rule of ordered) {
        const include = Array.isArray(rule?.match_any) ? rule.match_any : [];
        const exclude = Array.isArray(rule?.exclude_any) ? rule.exclude_any : [];
        const matches = include.some((token) => lowered.includes(String(token).toLowerCase()));
        const blocked = exclude.some((token) => lowered.includes(String(token).toLowerCase()));
        if (matches && !blocked) {
            return rule;
        }
    }
    return null;
}

function hasWidget(node, widgetName) {
    return Boolean(findWidget(node, widgetName));
}

function formatSetting(fieldName, value) {
    if (typeof value !== "number") {
        return null;
    }
    if (fieldName === "ctx_min") {
        return `ctx>=${value}`;
    }
    return `${fieldName}=${value}`;
}

function buildModeSummary(node, settings) {
    const parts = [];
    for (const fieldName of RECOMMENDATION_FIELDS) {
        if (!(fieldName in (settings || {}))) {
            continue;
        }
        if (fieldName === "ctx_min") {
            if (!hasWidget(node, "ctx")) {
                continue;
            }
        } else if (fieldName === "top_k") {
            if (!hasWidget(node, "top_k")) {
                continue;
            }
        } else if (!hasWidget(node, fieldName)) {
            continue;
        }
        const formatted = formatSetting(fieldName, settings[fieldName]);
        if (formatted) {
            parts.push(formatted);
        }
    }
    return parts;
}

function buildRecommendationText(node, rules) {
    const modelName = String(findWidget(node, "model_name")?.value || "").trim();
    if (!modelName || modelName.startsWith("(no ")) {
        return RECOMMENDATION_PLACEHOLDER;
    }

    const rule = matchRecommendationRule(modelName, rules);
    if (!rule) {
        return [
            "Manual settings guide",
            `No curated recommendation is bundled yet for ${modelName}.`,
            "Your current widget values stay exactly as you set them.",
        ].join("\n");
    }

    const thinkingEnabled = Boolean(findWidget(node, "enable_thinking")?.value);
    const activeSettings = thinkingEnabled ? rule.thinking_on : rule.thinking_off;
    const alternateSettings = thinkingEnabled ? rule.thinking_off : rule.thinking_on;
    const activeSummary = buildModeSummary(node, activeSettings);
    const alternateSummary = buildModeSummary(node, alternateSettings);
    const scope = recommendationScope(node);
    const lines = [
        rule.family_label || "Manual settings guide",
        `Current toggle: ${thinkingEnabled ? "thinking ON" : "thinking OFF"}`,
    ];

    const supportNote = rule.support_notes?.[scope];
    if (supportNote) {
        lines.push(`Toggle: ${supportNote}`);
    }
    lines.push(`Thinking ON: ${(thinkingEnabled ? activeSummary : alternateSummary).join(" | ") || "use the provider defaults"}`);
    lines.push(`Thinking OFF: ${(thinkingEnabled ? alternateSummary : activeSummary).join(" | ") || "use the provider defaults"}`);

    const softSwitch = rule.support_notes?.soft_switch;
    if (softSwitch) {
        lines.push(`Note: ${softSwitch}`);
    }
    if (Array.isArray(rule.notes) && rule.notes.length) {
        lines.push(`Extra: ${rule.notes[0]}`);
    }
    if (Array.isArray(rule.sources) && rule.sources.length) {
        lines.push(`Sources: ${rule.sources.join(" + ")}`);
    }
    return lines.join("\n");
}

function ensureRecommendationWidget(node) {
    if (!THINKINGLLM_NODE_NAMES.has(node.comfyClass)) {
        return null;
    }
    let widget = findWidget(node, RECOMMENDATION_WIDGET_NAME);
    if (widget) {
        return widget;
    }
    widget = ComfyWidgets.STRING(node, RECOMMENDATION_WIDGET_NAME, ["STRING", { multiline: true }], app).widget;
    if (widget?.inputEl) {
        widget.inputEl.readOnly = true;
        widget.inputEl.style.opacity = 0.78;
        widget.inputEl.style.fontSize = "12px";
        widget.inputEl.style.lineHeight = "1.35";
        widget.inputEl.rows = 6;
    }
    widget.options = { ...(widget.options || {}), serialize: false };
    widget.serializeValue = async () => undefined;
    widget.value = RECOMMENDATION_PLACEHOLDER;
    return widget;
}

function resizeNode(node) {
    requestAnimationFrame(() => {
        const size = node.computeSize();
        if (size[0] < node.size[0]) {
            size[0] = node.size[0];
        }
        if (size[1] < node.size[1]) {
            size[1] = node.size[1];
        }
        node.setSize(size);
        app.graph?.setDirtyCanvas?.(true, false);
    });
}

function updateRecommendationWidget(node) {
    const widget = ensureRecommendationWidget(node);
    if (!widget) {
        return;
    }
    loadRecommendations().then((rules) => {
        widget.value = buildRecommendationText(node, rules);
        resizeNode(node);
    });
}

function queueRecommendationRefresh(node) {
    if (!node || node._thinkingllmRecommendationQueued) {
        return;
    }
    node._thinkingllmRecommendationQueued = true;
    requestAnimationFrame(() => {
        node._thinkingllmRecommendationQueued = false;
        updateRecommendationWidget(node);
    });
}

function hookRecommendationUpdates(node) {
    if (!THINKINGLLM_NODE_NAMES.has(node.comfyClass) || node._thinkingllmRecommendationHooked) {
        return;
    }
    node._thinkingllmRecommendationHooked = true;
    ensureRecommendationWidget(node);

    for (const widgetName of ["model_name", "enable_thinking"]) {
        const target = findWidget(node, widgetName);
        if (!target || target._thinkingllmRecommendationCallbackHooked) {
            continue;
        }
        target._thinkingllmRecommendationCallbackHooked = true;
        const originalCallback = target.callback;
        target.callback = function (...args) {
            const result = originalCallback?.apply(this, args);
            queueRecommendationRefresh(node);
            return result;
        };
    }

    const originalConfigure = node.onConfigure;
    node.onConfigure = function (...args) {
        const result = originalConfigure?.apply(this, args);
        queueRecommendationRefresh(this);
        return result;
    };

    queueRecommendationRefresh(node);
}

const ext = {
    name: "QwenVL.appearance",

    nodeCreated(node) {
        const nclass = node.comfyClass;
        if (NODE_COLORS.hasOwnProperty(nclass)) {
            const colorKey = NODE_COLORS[nclass];
            const theme = COLOR_THEMES[colorKey];
            setNodeColors(node, theme);
        }
        hideGgufAdvancedInternals(node);
        hookRecommendationUpdates(node);
        clearHfTokenAfterExecution(node);
    }
};

app.registerExtension(ext);