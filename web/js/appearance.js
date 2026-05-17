import { app } from "/scripts/app.js";

const COLOR_THEMES = {
    QwenVL: { nodeColor: "#28403f", nodeBgColor: "#374539", width: 340 },
    QwenVLGGUF: { nodeColor: "#474539", nodeBgColor: "#2c4045", width: 340 },
    Tools: { nodeColor: "#28403f", nodeBgColor: "#233238", width: 300 },
    Enhancer: { nodeColor: "#374445", nodeBgColor: "#474539", width: 340 },
};

const NODE_COLORS = {
    // QwenVL nodes
    "AILab_QwenVL": "QwenVL",
    "AILab_QwenVL_Advanced": "QwenVL",
    "AILab_QwenVL_PromptEnhancer": "Enhancer",
    "AILab_QwenVL_GGUF": "QwenVLGGUF",
    "AILab_QwenVL_GGUF_Advanced": "QwenVLGGUF",
    "AILab_QwenVL_GGUF_PromptEnhancer": "Enhancer",

    // Tools
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

const ext = {
    name: "QwenVL.appearance",

    nodeCreated(node) {
        const nclass = node.comfyClass;
        if (NODE_COLORS.hasOwnProperty(nclass)) {
            let colorKey = NODE_COLORS[nclass];
            const theme = COLOR_THEMES[colorKey];
            setNodeColors(node, theme);
        }
        hideGgufAdvancedInternals(node);
        clearHfTokenAfterExecution(node);
    }
};

app.registerExtension(ext);