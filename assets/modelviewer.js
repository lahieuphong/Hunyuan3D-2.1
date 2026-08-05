(() => {
    "use strict";

    const MODE_ORDER = ["original", "white", "wireframe"];
    const MODE_MESSAGE_KEYS = {
        original: "modeOriginal",
        white: "modeWhite",
        wireframe: "modeWireframe",
    };
    const DEFAULT_LOCALE = "en";
    const LOCALE_STORAGE_KEY = "hunyuan3d.ui-locale.v1";
    const normalizeLocale = (value) => {
        if (typeof value !== "string") {
            return null;
        }
        const candidate = value.trim().toLowerCase().replace("_", "-");
        if (candidate === "zh" || candidate === "zh-cn" || candidate === "zh-hans") {
            return "zh-CN";
        }
        if (candidate === "en" || candidate === "en-us") {
            return "en";
        }
        return null;
    };
    const DEFAULT_CAMERA_ORBIT = "0deg 75deg 160%";
    const GROUND_TARGET_OFFSET_RATIO = 0;

    const iconPaths = {
        box: '<path d="m3 8 9-5 9 5-9 5-9-5Z"></path><path d="m3 8v8l9 5 9-5V8"></path><path d="M12 13v8"></path>',
        palette: '<path d="M12 3a9 9 0 0 0 0 18h1.2a1.8 1.8 0 0 0 1.2-3.1 1.8 1.8 0 0 1 1.2-3.1H18A3 3 0 0 0 21 12a9 9 0 0 0-9-9Z"></path><circle cx="7.5" cy="10" r=".8"></circle><circle cx="10" cy="6.8" r=".8"></circle><circle cx="14" cy="6.8" r=".8"></circle><circle cx="16.5" cy="10" r=".8"></circle>',
        wireframe: '<path d="m3 8 9-5 9 5-9 5-9-5Z"></path><path d="m3 8v8l9 5 9-5V8"></path><path d="M12 13v8M3 16l9-3 9 3M7.5 5.5 12 13l4.5-7.5"></path>',
        maximize: '<path d="M8 3H3v5"></path><path d="m3 3 6 6"></path><path d="M16 3h5v5"></path><path d="m21 3-6 6"></path><path d="M8 21H3v-5"></path><path d="m3 21 6-6"></path><path d="M16 21h5v-5"></path><path d="m21 21-6-6"></path>',
        reset: '<path d="M3 12a9 9 0 1 0 3-6.7"></path><path d="M3 4v6h6"></path>',
        rotate: '<path d="M21 12a9 9 0 0 1-15.2 6.5"></path><path d="M3 12A9 9 0 0 1 18.2 5.5"></path><path d="M18 2v4h4"></path><path d="M6 22v-4H2"></path>',
        grid: '<rect x="3" y="3" width="18" height="18" rx="2"></rect><path d="M3 9h18M3 15h18M9 3v18M15 3v18"></path>',
    };

    const iconMarkup = (name) => {
        const paths = iconPaths[name];
        return paths
            ? '<svg class="viewer-icon" viewBox="0 0 24 24" aria-hidden="true">' + paths + "</svg>"
            : "";
    };

    document.querySelectorAll("[data-icon]").forEach((element) => {
        if (!element.querySelector(".viewer-icon")) {
            element.insertAdjacentHTML("afterbegin", iconMarkup(element.dataset.icon));
        }
    });

    const shell = document.getElementById("viewer-shell");
    const modelViewer = document.getElementById("modelviewer");
    const modeStrip = document.getElementById("viewer-mode-strip");
    const loading = document.getElementById("viewer-loading");
    const loadingLabel = document.getElementById("viewer-loading-label");
    const status = document.getElementById("viewer-status");
    const toolbar = document.querySelector(".viewer-toolbar");
    const fullscreenButton = document.querySelector('[data-action="fullscreen"]');
    const resetButton = document.querySelector('[data-action="reset"]');
    const rotateButton = document.querySelector('[data-action="rotate"]');
    const gridButton = document.querySelector('[data-action="grid"]');
    const modeButtons = new Map();

    document.querySelectorAll(".viewer-mode[data-view-mode]").forEach((button) => {
        if (MODE_ORDER.includes(button.dataset.viewMode)) {
            modeButtons.set(button.dataset.viewMode, button);
        }
    });

    if (!shell || !modelViewer || !modeStrip) {
        return;
    }

    let parsedConfig = {};
    let configError = false;
    try {
        const configElement = document.getElementById("viewer-config");
        parsedConfig = JSON.parse(configElement ? configElement.textContent || "{}" : "{}");
    } catch (_error) {
        configError = true;
    }

    const messageCatalog = (
        parsedConfig
        && parsedConfig.messages
        && typeof parsedConfig.messages === "object"
    ) ? parsedConfig.messages : {};
    const parentLocale = () => {
        if (window.parent === window) {
            return null;
        }
        try {
            if (typeof window.parent.currentUiLocale === "function") {
                return normalizeLocale(window.parent.currentUiLocale());
            }
            return normalizeLocale(window.parent.document.documentElement.lang);
        } catch (_error) {
            return null;
        }
    };
    const queryLocale = () => {
        try {
            return normalizeLocale(new URL(window.location.href).searchParams.get("lang"));
        } catch (_error) {
            return null;
        }
    };
    const storedLocale = () => {
        try {
            return normalizeLocale(window.localStorage.getItem(LOCALE_STORAGE_KEY));
        } catch (_error) {
            return null;
        }
    };
    let currentLocale = (
        parentLocale()
        || queryLocale()
        || storedLocale()
        || normalizeLocale(parsedConfig.locale)
        || DEFAULT_LOCALE
    );
    const message = (key, parameters = {}) => {
        const localeMessages = messageCatalog[currentLocale] || {};
        const englishMessages = messageCatalog[DEFAULT_LOCALE] || {};
        const template = localeMessages[key] || englishMessages[key] || key;
        return Object.entries(parameters).reduce(
            (result, [name, value]) => {
                const displayValue = MODE_ORDER.includes(value)
                    ? modeLabel(value)
                    : value;
                return result.replaceAll("{" + name + "}", String(displayValue));
            },
            template
        );
    };
    const modeLabel = (mode) => message(MODE_MESSAGE_KEYS[mode] || "requestedMode");

    const variants = {};
    MODE_ORDER.forEach((mode) => {
        const candidate = parsedConfig
            && parsedConfig.variants
            && parsedConfig.variants[mode];
        if (candidate && typeof candidate.src === "string" && candidate.src.trim()) {
            variants[mode] = {src: candidate.src.trim()};
        }
    });

    const failedModes = new Set();
    const state = {
        activeMode: null,
        busy: false,
        cameraSnapshot: null,
        modelLoaded: false,
        notice: null,
        pendingMode: null,
        statusKey: "",
        statusParameters: {},
        statusTimer: null,
    };

    const absoluteSource = (source) => {
        try {
            return new URL(source, document.baseURI).href;
        } catch (_error) {
            return source;
        }
    };

    const resolveCurrentMode = () => {
        const source = absoluteSource(modelViewer.getAttribute("src") || "");
        return MODE_ORDER.find((mode) => (
            variants[mode] && absoluteSource(variants[mode].src) === source
        )) || null;
    };

    const renderStatus = () => {
        const text = state.statusKey
            ? message(state.statusKey, state.statusParameters)
            : "";
        status.textContent = text;
        status.hidden = !text;
    };

    const showStatus = (
        key,
        parameters = {},
        kind = "info",
        autoHide = false
    ) => {
        window.clearTimeout(state.statusTimer);
        state.statusKey = key;
        state.statusParameters = parameters;
        status.dataset.kind = kind;
        renderStatus();
        if (key && autoHide) {
            state.statusTimer = window.setTimeout(() => {
                state.statusKey = "";
                state.statusParameters = {};
                status.hidden = true;
                status.textContent = "";
            }, 1600);
        }
    };

    const updateModeButtons = () => {
        MODE_ORDER.forEach((mode) => {
            const button = modeButtons.get(mode);
            if (!button) {
                return;
            }
            const unavailable = !variants[mode] || failedModes.has(mode);
            const disabled = unavailable || state.busy;
            const active = state.activeMode === mode && !state.busy;
            button.disabled = disabled;
            button.classList.toggle("is-active", active);
            button.setAttribute("aria-disabled", String(disabled));
            button.setAttribute("aria-pressed", String(active));
            if (failedModes.has(mode)) {
                button.title = message("modeFailedTitle", {mode});
            } else if (!variants[mode]) {
                button.title = message("modeUnavailableTitle", {mode});
            } else {
                button.title = message("showModeTitle", {mode});
            }
        });
        modeStrip.setAttribute("aria-busy", String(state.busy));
    };

    const setLoading = (busy, mode = null) => {
        state.busy = busy;
        loading.hidden = !busy;
        shell.dataset.loading = String(busy);
        modelViewer.setAttribute("aria-busy", String(busy));
        if (busy && mode) {
            loadingLabel.textContent = message("loadingMode", {mode});
        } else if (!busy) {
            loadingLabel.textContent = message("loadingModel");
        }
        updateModeButtons();
    };

    const applyLocale = (requestedLocale) => {
        const nextLocale = normalizeLocale(requestedLocale) || DEFAULT_LOCALE;
        currentLocale = nextLocale;
        document.documentElement.lang = nextLocale;
        document.title = message("title");
        shell.setAttribute("aria-label", message("modelPreview"));
        modelViewer.setAttribute("alt", message("modelPreview"));
        try {
            if (window.frameElement) {
                window.frameElement.title = message("modelPreview");
            }
        } catch (_error) {
            // A cross-origin parent owns its iframe title.
        }
        toolbar?.setAttribute("aria-label", message("viewerControls"));
        modeStrip.setAttribute("aria-label", message("displayModes"));

        document.querySelectorAll("[data-viewer-message]").forEach((element) => {
            element.textContent = message(element.dataset.viewerMessage);
        });
        document.querySelectorAll("[data-viewer-mode-label]").forEach((element) => {
            element.textContent = modeLabel(element.dataset.viewerModeLabel);
        });

        [
            [fullscreenButton, "fullscreen"],
            [resetButton, "resetCamera"],
            [rotateButton, "toggleAutoRotate"],
            [gridButton, "toggleFloorGrid"],
        ].forEach(([button, key]) => {
            if (!button) {
                return;
            }
            const label = message(key);
            button.setAttribute("aria-label", label);
            button.title = label;
        });

        if (state.busy && state.pendingMode) {
            loadingLabel.textContent = message("loadingMode", {
                mode: state.pendingMode,
            });
        } else {
            loadingLabel.textContent = message("loadingModel");
        }
        updateModeButtons();
        renderStatus();
    };

    const finiteNumber = (value) => (
        typeof value === "number" && Number.isFinite(value)
    );

    const captureCameraState = () => {
        let cameraOrbit = modelViewer.getAttribute("camera-orbit") || DEFAULT_CAMERA_ORBIT;
        let cameraTarget = modelViewer.getAttribute("camera-target") || "auto auto auto";
        const orbit = modelViewer.getCameraOrbit && modelViewer.getCameraOrbit();
        const target = modelViewer.getCameraTarget && modelViewer.getCameraTarget();

        if (
            orbit
            && finiteNumber(orbit.theta)
            && finiteNumber(orbit.phi)
            && finiteNumber(orbit.radius)
        ) {
            cameraOrbit = orbit.theta + "rad " + orbit.phi + "rad " + orbit.radius + "m";
        }
        if (
            target
            && finiteNumber(target.x)
            && finiteNumber(target.y)
            && finiteNumber(target.z)
        ) {
            cameraTarget = target.x + "m " + target.y + "m " + target.z + "m";
        }

        return {
            autoRotate: modelViewer.hasAttribute("auto-rotate"),
            cameraOrbit,
            cameraTarget,
            gridVisible: !shell.classList.contains("grid-hidden"),
        };
    };

    const restoreViewerState = (snapshot) => {
        if (!snapshot) {
            return;
        }
        modelViewer.setAttribute("camera-target", snapshot.cameraTarget);
        modelViewer.setAttribute("camera-orbit", snapshot.cameraOrbit);
        modelViewer.toggleAttribute("auto-rotate", snapshot.autoRotate);
        shell.classList.toggle("grid-hidden", !snapshot.gridVisible);
        rotateButton.setAttribute("aria-pressed", String(snapshot.autoRotate));
        gridButton.setAttribute("aria-pressed", String(snapshot.gridVisible));
        if (modelViewer.jumpCameraToGoal) {
            modelViewer.jumpCameraToGoal();
        }
    };

    const getGroundedCameraTarget = () => {
        const center = modelViewer.getBoundingBoxCenter && modelViewer.getBoundingBoxCenter();
        const dimensions = modelViewer.getDimensions && modelViewer.getDimensions();
        if (!center || !dimensions || !finiteNumber(dimensions.y) || dimensions.y <= 0) {
            return "auto auto auto";
        }

        const targetY = center.y - dimensions.y * GROUND_TARGET_OFFSET_RATIO;
        return center.x + "m " + targetY + "m " + center.z + "m";
    };

    const applyCamera = (orbit) => {
        modelViewer.setAttribute("camera-target", getGroundedCameraTarget());
        modelViewer.setAttribute("camera-orbit", orbit);
        if (modelViewer.jumpCameraToGoal) {
            modelViewer.jumpCameraToGoal();
        }
    };

    const resetCamera = () => {
        applyCamera(DEFAULT_CAMERA_ORBIT);
    };

    const finishModeLoad = () => {
        const loadedMode = state.pendingMode || resolveCurrentMode();
        modelViewer.setAttribute("environment-image", "/static/env_maps/gradient.jpg");
        const cameraSnapshot = state.cameraSnapshot;
        state.cameraSnapshot = null;
        if (cameraSnapshot) {
            restoreViewerState(cameraSnapshot);
        } else {
            resetCamera();
        }
        state.modelLoaded = true;
        state.activeMode = loadedMode;
        state.pendingMode = null;
        setLoading(false);

        if (state.notice) {
            showStatus(
                state.notice.key,
                state.notice.parameters,
                "error",
                false
            );
            state.notice = null;
        } else if (loadedMode) {
            showStatus(
                "modeReady",
                {mode: loadedMode},
                "info",
                true
            );
        }
    };

    const requestMode = (mode, options = {}) => {
        if (
            !MODE_ORDER.includes(mode)
            || !variants[mode]
            || failedModes.has(mode)
            || state.busy
        ) {
            return;
        }
        if (state.activeMode === mode && state.modelLoaded) {
            return;
        }

        state.cameraSnapshot = options.cameraSnapshot || (
            state.modelLoaded ? captureCameraState() : null
        );
        state.notice = options.notice || null;
        state.pendingMode = mode;
        setLoading(true, mode);
        showStatus("", {}, "info", false);

        const requestedSource = absoluteSource(variants[mode].src);
        const currentSource = absoluteSource(modelViewer.getAttribute("src") || "");
        if (requestedSource === currentSource && state.modelLoaded) {
            finishModeLoad();
            return;
        }
        if (requestedSource !== currentSource) {
            state.modelLoaded = false;
            modelViewer.setAttribute("src", variants[mode].src);
        }
    };

    const handleModelError = () => {
        const failedMode = state.pendingMode || resolveCurrentMode();
        if (failedMode) {
            failedModes.add(failedMode);
        }
        const cameraSnapshot = state.cameraSnapshot;
        state.modelLoaded = false;
        const failedLabel = failedMode || message("requestedMode");
        state.pendingMode = null;
        state.activeMode = null;
        setLoading(false);

        if (
            failedMode !== "white"
            && variants.white
            && !failedModes.has("white")
        ) {
            requestMode("white", {
                cameraSnapshot,
                notice: {
                    key: "fallbackMode",
                    parameters: {
                        mode: failedLabel,
                        fallback: "white",
                    },
                },
            });
            return;
        }

        showStatus("modeFailed", {mode: failedLabel}, "error", false);
        updateModeButtons();
    };

    modelViewer.addEventListener("load", finishModeLoad);
    modelViewer.addEventListener("error", handleModelError);

    fullscreenButton.addEventListener("click", async () => {
        try {
            if (document.fullscreenElement) {
                await document.exitFullscreen();
            } else {
                await shell.requestFullscreen();
            }
        } catch (_error) {
            showStatus("fullscreenUnavailable", {}, "error", true);
        }
    });

    resetButton.addEventListener("click", resetCamera);

    rotateButton.addEventListener("click", () => {
        const enabled = !modelViewer.hasAttribute("auto-rotate");
        modelViewer.toggleAttribute("auto-rotate", enabled);
        rotateButton.setAttribute("aria-pressed", String(enabled));
    });

    gridButton.addEventListener("click", () => {
        const visible = shell.classList.toggle("grid-hidden") === false;
        gridButton.setAttribute("aria-pressed", String(visible));
    });

    modeButtons.forEach((button, mode) => {
        button.addEventListener("click", () => requestMode(mode));
    });

    applyLocale(currentLocale);

    if (window.parent !== window) {
        try {
            window.parent.addEventListener("ui-language-change", (event) => {
                applyLocale(event?.detail?.locale || parentLocale());
            });
        } catch (_error) {
            // Cross-origin embeds retain their URL/configured locale.
        }
    }
    window.addEventListener("storage", (event) => {
        if (event.key === LOCALE_STORAGE_KEY) {
            applyLocale(event.newValue);
        }
    });

    if (configError) {
        showStatus("configurationUnreadable", {}, "error", false);
    }

    const configuredDefault = typeof parsedConfig.defaultMode === "string"
        ? parsedConfig.defaultMode
        : "";
    const initialMode = (
        MODE_ORDER.includes(configuredDefault)
        && variants[configuredDefault]
    )
        ? configuredDefault
        : variants.white
            ? "white"
            : MODE_ORDER.find((mode) => variants[mode]) || null;

    if (initialMode) {
        window.customElements.whenDefined("model-viewer").then(() => {
            requestMode(initialMode);
        });
    } else {
        updateModeButtons();
    }
})();
