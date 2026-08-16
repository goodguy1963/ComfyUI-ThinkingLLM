import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

const COLOR_THEMES = {
    QwenVL: { nodeColor: "#28403f", nodeBgColor: "#374539", width: 340 },
    QwenVLGGUF: { nodeColor: "#474539", nodeBgColor: "#2c4045", width: 340 },
    Tools: { nodeColor: "#28403f", nodeBgColor: "#233238", width: 300 },
    Enhancer: { nodeColor: "#374445", nodeBgColor: "#474539", width: 340 },
    Api: { nodeColor: "#3a2f4a", nodeBgColor: "#2b2740", width: 360 },
};

const NODE_COLORS = {
    "AILab_QwenVL": "QwenVL",
    "AILab_QwenVL_Advanced": "QwenVL",
    "AILab_QwenVL_PromptEnhancer": "Enhancer",
    "AILab_QwenVL_GGUF": "QwenVLGGUF",
    "AILab_QwenVL_GGUF_Advanced": "QwenVLGGUF",
    "AILab_QwenVL_GGUF_PromptEnhancer": "Enhancer",
    "ThinkingLLM_QwenVL": "QwenVL",
    "ThinkingLLM_QwenVL_Advanced": "QwenVL",
    "ThinkingLLM_QwenVL_PromptEnhancer": "Enhancer",
    "ThinkingLLM_QwenVL_GGUF": "QwenVLGGUF",
    "ThinkingLLM_QwenVL_GGUF_Advanced": "QwenVLGGUF",
    "ThinkingLLM_Gemma4_Audio_GGUF": "QwenVLGGUF",
    "ThinkingLLM_Whisper_ASR": "QwenVLGGUF",
    "ThinkingLLM_QwenVL_GGUF_PromptEnhancer": "Enhancer",
    "ThinkingLLM_SystemPromptPreset": "Tools",
    "AILab_QwenVL_PromptLibrary": "Tools",
    "VRAMCleanup": "Tools",
    "StorySplitNode": "Tools",
    "ThinkingLLM_OpenAICompatibleAPI": "Api",
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
    "ThinkingLLM_Gemma4_Audio_GGUF",
    "ThinkingLLM_Whisper_ASR",
    "ThinkingLLM_QwenVL_GGUF_PromptEnhancer",
]);

const PRESET_TOOLTIP_NODE_NAMES = new Set([
    ...THINKINGLLM_NODE_NAMES,
    "ThinkingLLM_SystemPromptPreset",
]);

const DURATION_INPUT_NODE_NAMES = new Set([
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

const AUDIO_CAPABLE_NODE_NAMES = new Set([
    "ThinkingLLM_Gemma4_Audio_GGUF",
    "ThinkingLLM_Whisper_ASR",
    "AILab_QwenVL_GGUF",
    "AILab_QwenVL_GGUF_Advanced",
    "ThinkingLLM_QwenVL_GGUF",
    "ThinkingLLM_QwenVL_GGUF_Advanced",
]);

const GGUF_ADVANCED_INTERNAL_WIDGETS = new Set([
    "legacy_seed_mode",
    "legacy_unload_after_run",
]);

const RECOMMENDATIONS_URL = new URL("../model_recommendations.json", import.meta.url);
const PRESET_TOOLTIPS_URL = new URL("../preset_tooltips.json", import.meta.url);
const API_MODEL_CATALOGS_URL = new URL("../api_model_catalogs.json", import.meta.url);
const API_NODE_CLASS = "ThinkingLLM_OpenAICompatibleAPI";
const API_PROFILE_WIDGET = "api_profile";
const API_MODEL_WIDGET = "model_name";
const API_HELP_WIDGET_NAME = "setup_help";
const RECOMMENDATION_WIDGET_NAME = "recommended_settings";
const RECOMMENDATION_PLACEHOLDER = "Select a model to see provider/community recommended settings. This note never changes your saved widget values.";
const DEFAULT_DURATION_SECONDS = 5.0;
const RECOMMENDATION_FIELDS = ["temperature", "top_p", "top_k", "max_tokens", "ctx_min", "n_batch", "image_max_tokens"];
const AUDIO_UNSUPPORTED_MESSAGE = "No curated audio support for the selected model. Use Gemma 4 Audio for audio understanding or Whisper ASR for transcription.";
const PRESET_WIDGET_NAMES = ["preset_prompt", "enhancement_style", "preset_system_prompt"];

let recommendationsPromise = null;
let presetTooltipsPromise = null;
let apiModelCatalogsPromise = null;

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

function loadApiModelCatalogs() {
    if (!apiModelCatalogsPromise) {
        apiModelCatalogsPromise = fetch(API_MODEL_CATALOGS_URL, { cache: "no-store" })
            .then((response) => {
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                return response.json();
            })
            .then((payload) => (payload?.profiles && typeof payload.profiles === "object" ? payload.profiles : {}))
            .catch((error) => {
                console.warn("[ThinkingLLM] Failed to load API model catalogs", error);
                return {};
            });
    }
    return apiModelCatalogsPromise;
}

function loadPresetTooltips() {
    if (!presetTooltipsPromise) {
        presetTooltipsPromise = fetch(PRESET_TOOLTIPS_URL, { cache: "no-store" })
            .then((response) => {
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}`);
                }
                return response.json();
            })
            .catch((error) => {
                console.warn("[ThinkingLLM] Failed to load preset tooltips", error);
                return {};
            });
    }
    return presetTooltipsPromise;
}

function setNodeColors(node, theme) {
    if (!theme) { return; }
    if (!node.color && theme.nodeColor) {
        node.color = theme.nodeColor;
    }
    if (!node.bgcolor && theme.nodeBgColor) {
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
    widget.computeSize = () => [0, 0];
    widget.draw = () => {};
}

function setWidgetVisible(node, widget, visible) {
    if (!widget) { return; }
    if (!widget._thinkingllmVisibilityState) {
        widget._thinkingllmVisibilityState = {
            computeSize: widget.computeSize,
            hidden: widget.hidden,
        };
    }
    const original = widget._thinkingllmVisibilityState;
    if (visible) {
        widget.computeSize = original.computeSize;
        widget.hidden = original.hidden;
        delete widget.computedHeight;
    } else {
        widget.hidden = true;
        widget.computeSize = () => [0, -3.3];
        widget.computedHeight = 0;
    }
    resizeNode(node);
}

function repairDurationWidgetValue(widget) {
    if (!widget) { return; }
    const numericValue = typeof widget.value === "number"
        ? widget.value
        : Number(String(widget.value ?? "").trim());
    const minimum = Number(widget.options?.min ?? 0.2);
    const maximum = Number(widget.options?.max ?? 150.0);
    if (Number.isFinite(numericValue) && numericValue >= minimum && numericValue <= maximum) {
        widget.value = numericValue;
        widget._thinkingllmLastValidDuration = numericValue;
        return;
    }
    const previousValue = Number(widget._thinkingllmLastValidDuration);
    widget.value = Number.isFinite(previousValue) ? previousValue : DEFAULT_DURATION_SECONDS;
}

function readComboValue(widget) {
    const value = widget?.value;
    if (typeof value === "string") {
        return value;
    }
    if (value && typeof value === "object") {
        for (const key of ["value", "content", "label", "name"]) {
            if (typeof value[key] === "string") {
                return value[key];
            }
        }
    }
    return value == null ? "" : String(value);
}

function normalizePresetName(value) {
    return String(value || "").normalize("NFC").replaceAll("\uFE0F", "").trim();
}

function findVideoPresetMetadata(payload, selectedPreset) {
    const presets = payload?.video_presets;
    if (!presets || typeof presets !== "object" || Array.isArray(presets)) {
        return { metadataAvailable: false, metadata: null };
    }
    if (Object.prototype.hasOwnProperty.call(presets, selectedPreset)) {
        return { metadataAvailable: true, metadata: presets[selectedPreset] };
    }
    const normalizedSelection = normalizePresetName(selectedPreset);
    const matchingEntry = Object.entries(presets)
        .find(([name]) => normalizePresetName(name) === normalizedSelection);
    return {
        metadataAvailable: true,
        metadata: matchingEntry?.[1] || null,
    };
}

function selectedPresetName(node) {
    return PRESET_WIDGET_NAMES
        .map((name) => readComboValue(findWidget(node, name)))
        .find((value) => value) || "";
}

function updateDurationVisibility(node, payload) {
    if (!DURATION_INPUT_NODE_NAMES.has(node.comfyClass)) {
        return;
    }
    const selectedPreset = selectedPresetName(node);
    const durationWidget = findWidget(node, "duration_seconds");
    repairDurationWidgetValue(durationWidget);
    const { metadataAvailable, metadata } = findVideoPresetMetadata(payload, selectedPreset);
    if (!metadataAvailable) {
        // Keep the backend-provided control usable if the auxiliary payload is stale or unavailable.
        setWidgetVisible(node, durationWidget, true);
        return;
    }
    setWidgetVisible(node, durationWidget, Boolean(metadata?.duration_required));
}

function findWidgets(node, name) {
    return node.widgets?.filter((widget) => widget?.name === name) || [];
}

function removeDuplicateRecommendationWidgets(node, keepWidget) {
    if (!node.widgets?.length || !keepWidget) {
        return;
    }
    for (let index = node.widgets.length - 1; index >= 0; index -= 1) {
        const widget = node.widgets[index];
        if (widget?.name !== RECOMMENDATION_WIDGET_NAME || widget === keepWidget) {
            continue;
        }
        hideStableWidget(node, widget);
        node.widgets.splice(index, 1);
    }
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
        "ThinkingLLM_Gemma4_Audio_GGUF",
        "ThinkingLLM_Whisper_ASR",
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

function hasAudioInput(node) {
    return AUDIO_CAPABLE_NODE_NAMES.has(node.comfyClass)
        || Boolean(node.inputs?.some((input) => input?.name === "audio"));
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
    if (node.comfyClass === "ThinkingLLM_Whisper_ASR") {
        const modelSize = String(findWidget(node, "model_size")?.value || "small").trim() || "small";
        return [
            "Whisper ASR transcription",
            `Model: ${modelSize}`,
            "Audio: Use for speech-to-text from AUDIO or audio_file_path. M4A/MP3/WAV/FLAC are decoded with FFmpeg.",
            "Settings: small + CPU/int8 is the reliable Windows default; large-v3 is higher quality; vad_filter skips silence.",
            "Dependency: faster-whisper must be installed in the active ComfyUI Python.",
        ].join("\n");
    }

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
    const audioNote = rule.support_notes?.audio;
    if (audioNote && hasAudioInput(node)) {
        lines.push(`Audio: ${audioNote}`);
    } else if (hasAudioInput(node)) {
        lines.push(`Audio: ${AUDIO_UNSUPPORTED_MESSAGE}`);
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

function buildVideoPresetRecommendationText(node, payload) {
    const presetName = selectedPresetName(node);
    if (!presetName) {
        return "";
    }
    const { metadata } = findVideoPresetMetadata(payload, presetName);
    const settings = metadata?.recommended_settings;
    if (!settings || typeof settings !== "object") {
        return "";
    }

    const suggested = [];
    if (hasWidget(node, "enable_thinking") && typeof settings.enable_thinking === "boolean") {
        suggested.push(`thinking ${settings.enable_thinking ? "ON" : "OFF"}`);
    }
    if (hasWidget(node, "max_tokens") && Number.isFinite(settings.max_tokens)) {
        suggested.push(`max_tokens=${settings.max_tokens}`);
    }
    if (!suggested.length) {
        return "";
    }

    const lines = [
        `Video preset: ${presetName}`,
        `Suggested: ${suggested.join(" | ")}`,
    ];
    if (metadata.recommendation_note) {
        lines.push(`Reason: ${metadata.recommendation_note}`);
    }
    lines.push("Sampler: follow the model/input-mode recommendation above.");
    lines.push("Info only: your saved widget values are not changed.");
    return lines.join("\n");
}

function ensureRecommendationWidget(node) {
    if (!THINKINGLLM_NODE_NAMES.has(node.comfyClass)) {
        return null;
    }
    let widget = findWidgets(node, RECOMMENDATION_WIDGET_NAME)
        .find((candidate) => candidate?.inputEl)
        || findWidget(node, RECOMMENDATION_WIDGET_NAME);
    if (!widget) {
        widget = ComfyWidgets.STRING(node, RECOMMENDATION_WIDGET_NAME, ["STRING", { multiline: true }], app).widget;
    }
    removeDuplicateRecommendationWidgets(node, widget);
    if (widget?.inputEl) {
        widget.inputEl.readOnly = true;
        widget.inputEl.style.opacity = 0.78;
        widget.inputEl.style.fontSize = "12px";
        widget.inputEl.style.lineHeight = "1.35";
        widget.inputEl.rows = 10;
    }
    widget.options = { ...(widget.options || {}), serialize: false };
    widget.serializeValue = async () => undefined;
    if (!widget.value) {
        widget.value = RECOMMENDATION_PLACEHOLDER;
    }
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
    Promise.all([loadRecommendations(), loadPresetTooltips()]).then(([rules, payload]) => {
        widget.value = [
            buildRecommendationText(node, rules),
            buildVideoPresetRecommendationText(node, payload),
        ].filter(Boolean).join("\n\n");
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

function catalogModelsForProfile(catalogs, profileName) {
    const entry = catalogs?.[profileName];
    const models = Array.isArray(entry?.models) ? entry.models : [];
    return models
        .map((model) => (model && typeof model.id === "string" ? model.id : ""))
        .filter(Boolean);
}

function isApiNode(node) {
    return node?.comfyClass === API_NODE_CLASS;
}

function applyApiModelCombo(node, catalogs) {
    if (!isApiNode(node)) {
        return;
    }
    const profileWidget = findWidget(node, API_PROFILE_WIDGET);
    const modelWidget = findWidget(node, API_MODEL_WIDGET);
    if (!profileWidget || !modelWidget) {
        return;
    }
    const profileName = readComboValue(profileWidget);
    const models = catalogModelsForProfile(catalogs, profileName);
    const options = models.length ? [...new Set(models)] : null;

    if (options) {
        // Always keep a native combo when a curated list exists for this profile.
        if (modelWidget.type !== "combo") {
            modelWidget.type = "combo";
            modelWidget.serialize = true;
        }
        if (Array.isArray(modelWidget.options?.values)) {
            const equal = options.length === modelWidget.options.values.length
                && options.every((option, index) => option === modelWidget.options.values[index]);
            if (!equal) {
                modelWidget.options.values = options;
            }
        }
        // Keep the current value if it is still a valid option; otherwise fall back to the first.
        const currentValue = readComboValue(modelWidget);
        if (!options.includes(currentValue)) {
            modelWidget.value = options[0];
            modelWidget.options.value = options[0];
        }
    }
    resizeNode(node);
}

function hookApiModelDropdown(node) {
    if (!isApiNode(node) || node._thinkingllmApiModelHooked) {
        return;
    }
    node._thinkingllmApiModelHooked = true;

    const profileWidget = findWidget(node, API_PROFILE_WIDGET);
    const modelWidget = findWidget(node, API_MODEL_WIDGET);

    loadApiModelCatalogs().then((catalogs) => {
        const refresh = () => applyApiModelCombo(node, catalogs);
        if (profileWidget) {
            const originalCallback = profileWidget.callback;
            profileWidget.callback = function (...args) {
                const result = originalCallback?.apply(this, args);
                refresh();
                return result;
            };
        }
        refresh();
    });
}

function buildApiHelpText() {
    return [
        "SETUP (einmalig):",
        "",
        "Windows - API-Key als User-Umgebungsvariable (PowerShell):",
        '  [System.Environment]::SetEnvironmentVariable("OPENROUTER_API_KEY", "sk-...", "User")',
        "  (Variablennamen je Provider: OPENROUTER_API_KEY, OPENAI_API_KEY,",
        "   DASHSCOPE_API_KEY, GROQ_API_KEY, ORCAROUTER_API_KEY, ...)",
        "",
        "Linux/macOS - export im Terminal vor dem Start:",
        '  export OPENROUTER_API_KEY="sk-..."',
        "",
        "Danach ComfyUI NEU starten. Der Key liegt nur auf dem Server/Rechner -",
        "nie im Workflow, im Profil-JSON oder in Git.",
        "",
        "Doku: docs/OPENAI_COMPATIBLE_API.md",
    ].join("\n");
}

function ensureApiHelpWidget(node) {
    if (!isApiNode(node)) {
        return null;
    }
    let widget = findWidget(node, API_HELP_WIDGET_NAME);
    if (!widget) {
        widget = ComfyWidgets.STRING(node, API_HELP_WIDGET_NAME, ["STRING", { multiline: true }], app).widget;
    }
    if (widget?.inputEl) {
        widget.inputEl.readOnly = true;
        widget.inputEl.style.opacity = 0.75;
        widget.inputEl.style.fontSize = "12px";
        widget.inputEl.style.lineHeight = "1.35";
        widget.inputEl.rows = 12;
    }
    widget.options = { ...(widget.options || {}), serialize: false };
    widget.serializeValue = async () => undefined;
    widget.value = buildApiHelpText();
    return widget;
}

function hookApiHelp(node) {
    if (!isApiNode(node) || node._thinkingllmApiHelpHooked) {
        return;
    }
    node._thinkingllmApiHelpHooked = true;
    ensureApiHelpWidget(node);
    resizeNode(node);
}

function summarizePresetTooltip(prompt) {
    const text = String(prompt || "").trim();
    if (!text) {
        return "No preset prompt is active. Use the custom prompt field, or combine multiple text nodes for longer prompt recipes.";
    }
    return text.length > 1800 ? `${text.slice(0, 1800).trim()}...` : text;
}

function updatePresetTooltip(widget, promptMap) {
    if (!widget) {
        return;
    }
    const prompt = promptMap?.[widget.name]?.[widget.value];
    const tooltip = summarizePresetTooltip(prompt);
    widget.options = { ...(widget.options || {}), tooltip };
    widget.tooltip = tooltip;
    if (widget.inputEl) {
        widget.inputEl.title = tooltip;
    }
}

function hookPresetTooltipPreviews(node) {
    if (!PRESET_TOOLTIP_NODE_NAMES.has(node.comfyClass) || node._thinkingllmPresetTooltipsHooked) {
        return;
    }
    node._thinkingllmPresetTooltipsHooked = true;

    loadPresetTooltips().then((promptMap) => {
        const refresh = () => {
            for (const widgetName of PRESET_WIDGET_NAMES) {
                updatePresetTooltip(findWidget(node, widgetName), promptMap);
            }
            updateDurationVisibility(node, promptMap);
        };

        for (const widgetName of PRESET_WIDGET_NAMES) {
            const target = findWidget(node, widgetName);
            if (!target || target._thinkingllmPresetTooltipCallbackHooked) {
                continue;
            }
            target._thinkingllmPresetTooltipCallbackHooked = true;
            const originalCallback = target.callback;
            target.callback = function (...args) {
                const result = originalCallback?.apply(this, args);
                refresh();
                return result;
            };
        }

        const originalConfigure = node.onConfigure;
        node.onConfigure = function (...args) {
            const result = originalConfigure?.apply(this, args);
            refresh();
            return result;
        };

        refresh();
    });
}

function hookRecommendationUpdates(node) {
    if (!THINKINGLLM_NODE_NAMES.has(node.comfyClass) || node._thinkingllmRecommendationHooked) {
        return;
    }
    node._thinkingllmRecommendationHooked = true;
    ensureRecommendationWidget(node);

    for (const widgetName of ["model_name", "enable_thinking", ...PRESET_WIDGET_NAMES]) {
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
        hookPresetTooltipPreviews(node);
        clearHfTokenAfterExecution(node);
        hookApiModelDropdown(node);
        hookApiHelp(node);
    }
};

app.registerExtension(ext);
