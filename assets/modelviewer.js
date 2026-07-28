(() => {
    "use strict";

    const MODE_ORDER = ["original", "white", "wireframe"];
    const MODE_LABELS = {
        original: "Original",
        white: "White",
        wireframe: "Wireframe",
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
        notice: "",
        pendingMode: null,
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

    const showStatus = (message, kind = "info", autoHide = false) => {
        window.clearTimeout(state.statusTimer);
        status.textContent = message;
        status.dataset.kind = kind;
        status.hidden = !message;
        if (message && autoHide) {
            state.statusTimer = window.setTimeout(() => {
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
                button.title = MODE_LABELS[mode] + " model could not be loaded";
            } else if (!variants[mode]) {
                button.title = MODE_LABELS[mode] + " model is unavailable";
            } else {
                button.title = "Show " + MODE_LABELS[mode] + " model";
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
            loadingLabel.textContent = "Loading " + MODE_LABELS[mode] + "…";
        }
        updateModeButtons();
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
            showStatus(state.notice, "error", false);
            state.notice = "";
        } else if (loadedMode) {
            showStatus(MODE_LABELS[loadedMode] + " model ready", "info", true);
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
        state.notice = options.notice || "";
        state.pendingMode = mode;
        setLoading(true, mode);
        showStatus("", "info", false);

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
        const failedLabel = MODE_LABELS[failedMode] || "Requested";
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
                notice: failedLabel + " is unavailable. Showing White instead.",
            });
            return;
        }

        showStatus(failedLabel + " model could not be loaded.", "error", false);
        updateModeButtons();
    };

    modelViewer.addEventListener("load", finishModeLoad);
    modelViewer.addEventListener("error", handleModelError);

    document.querySelector('[data-action="fullscreen"]').addEventListener("click", async () => {
        try {
            if (document.fullscreenElement) {
                await document.exitFullscreen();
            } else {
                await shell.requestFullscreen();
            }
        } catch (_error) {
            showStatus("Fullscreen is unavailable in this browser.", "error", true);
        }
    });

    document.querySelector('[data-action="reset"]').addEventListener("click", resetCamera);

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

    updateModeButtons();

    if (configError) {
        showStatus("Viewer configuration could not be read.", "error", false);
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
