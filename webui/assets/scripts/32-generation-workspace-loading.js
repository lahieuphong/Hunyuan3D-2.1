
        const generationWorkspaceLoadingHosts = Object.freeze({
            preview: "mesh-viewer-loading",
            export: "mesh-export-loading",
            statistics: "mesh-stats-loading",
        });
        const generationWorkspaceSurfaceIds = Object.freeze([
            "mesh-viewer",
            "mesh-export-viewer",
            "mesh-stats",
        ]);
        const generationWorkspaceTabIds = Object.freeze([
            "gen_mesh_panel",
            "export_mesh_panel",
            "stats_panel",
        ]);
        let generationWorkspaceLoadingState = {
            status: "idle",
            uid: null,
            progress: 0,
            message: "",
            resumed: false,
        };
        let generationWorkspaceReadyObserver = null;
        let generationWorkspaceReadyTimer = null;
        let generationWorkspaceCompleteTimer = null;

        const generationWorkspaceSafeProgress = (value) => {
            const progress = Number(value);
            return Number.isFinite(progress)
                ? Math.max(0, Math.min(100, progress))
                : 0;
        };

        const setGenerationWorkspaceText = (element, value) => {
            if (!element) return;
            const nextValue = String(value ?? "");
            if (element.textContent !== nextValue) element.textContent = nextValue;
        };

        const generationWorkspaceKindTitle = (kind) => {
            if (kind === "export") {
                return uiT("generation.loading.export_title");
            }
            if (kind === "statistics") {
                return uiT("generation.loading.statistics_title");
            }
            return uiT("generation.loading.preview_title");
        };

        const generationWorkspaceKindDescription = (kind) => {
            if (kind === "export") {
                return uiT("generation.loading.export_body");
            }
            if (kind === "statistics") {
                return uiT("generation.loading.statistics_body");
            }
            return uiT("generation.loading.preview_body");
        };

        const generationWorkspacePhase = (state) => {
            if (state.status === "failed") {
                return uiT("generation.loading.phase.failed");
            }
            if (state.status === "complete") {
                return uiT("generation.loading.phase.complete");
            }
            if (state.status === "finalizing") {
                return uiT("generation.loading.phase.finalizing");
            }
            if (state.status === "queued") {
                return state.resumed
                    ? uiT("generation.loading.phase.restoring")
                    : uiT("generation.loading.phase.queued");
            }
            if (state.progress < 30) {
                return uiT("generation.loading.phase.inputs");
            }
            if (state.progress < 75) {
                return uiT("generation.loading.phase.geometry");
            }
            if (state.progress < 94) {
                return uiT("generation.loading.phase.mesh");
            }
            if (state.progress < 99) {
                return uiT("generation.loading.phase.asset");
            }
            return uiT("generation.loading.phase.preview");
        };

        const generationWorkspaceDisplayTitle = (kind, state) => {
            if (state.status === "failed") {
                return uiT("generation.loading.failed_title");
            }
            if (state.status === "complete") {
                return uiT("generation.loading.complete_title");
            }
            if (state.status === "finalizing") {
                return uiT("generation.loading.finalizing_title");
            }
            return generationWorkspaceKindTitle(kind);
        };

        const generationWorkspaceDisplayMessage = (kind, state) => {
            if (state.status === "failed") {
                return uiT("generation.loading.failed_body");
            }
            if (state.status === "complete") {
                return uiT("generation.loading.complete_body");
            }
            if (state.status === "finalizing") {
                return uiT("generation.loading.finalizing_body");
            }
            return state.message || generationWorkspaceKindDescription(kind);
        };

        const generationWorkspaceIsBusy = (status) => (
            status === "queued"
            || status === "running"
            || status === "finalizing"
        );

        const clearGenerationWorkspaceReadyWait = () => {
            generationWorkspaceReadyObserver?.disconnect();
            generationWorkspaceReadyObserver = null;
            if (generationWorkspaceReadyTimer !== null) {
                window.clearInterval(generationWorkspaceReadyTimer);
                generationWorkspaceReadyTimer = null;
            }
        };

        const clearGenerationWorkspaceCompleteTimer = () => {
            if (generationWorkspaceCompleteTimer !== null) {
                window.clearTimeout(generationWorkspaceCompleteTimer);
                generationWorkspaceCompleteTimer = null;
            }
        };

        const installGenerationWorkspaceTabIndicators = () => {
            const outputTabs = document.getElementById("output-tabs");
            if (!outputTabs) return;
            generationWorkspaceTabIds.forEach((tabId) => {
                const button = outputTabs.querySelector(
                    'button[role="tab"][data-tab-id="' + tabId + '"]'
                );
                if (!button || button.querySelector(".generation-tab-loading-indicator")) {
                    return;
                }
                const indicator = document.createElement("span");
                indicator.className = "generation-tab-loading-indicator";
                indicator.setAttribute("aria-hidden", "true");
                button.append(indicator);
            });
        };

        const installGenerationActionSpinner = () => {
            const button = document.getElementById("generate-3d-button");
            if (!button || button.querySelector(".generation-action-spinner")) return;
            const spinner = document.createElement("span");
            spinner.className = "generation-action-spinner";
            spinner.setAttribute("aria-hidden", "true");
            button.insertBefore(spinner, button.firstChild);
        };

        const syncGenerationSidebarLoading = (busy) => {
            const consolePanel = document.getElementById("generation-console-panel");
            consolePanel?.classList.toggle("is-generation-running", busy);
            consolePanel?.classList.toggle(
                "is-generation-failed",
                generationWorkspaceLoadingState.status === "failed"
            );
            consolePanel?.setAttribute("aria-busy", busy ? "true" : "false");
            document.getElementById("generation-output-card")?.setAttribute(
                "aria-busy",
                busy ? "true" : "false"
            );
            document.querySelectorAll(".generation-detail strong").forEach((element) => {
                const copy = String(element.textContent || "").trim();
                element.classList.toggle(
                    "is-generation-metric-pending",
                    busy && (!copy || copy === "-" || copy === "\u2014")
                );
            });

            const outputMeta = document.getElementById("generation-output-meta");
            if (!outputMeta) return;
            if (generationWorkspaceLoadingState.status === "queued") {
                setGenerationWorkspaceText(
                    outputMeta,
                    uiT("generation.loading.output_file")
                );
            } else if (generationWorkspaceLoadingState.status === "finalizing") {
                setGenerationWorkspaceText(
                    outputMeta,
                    uiT("generation.loading.output_finalizing")
                );
            } else if (generationWorkspaceLoadingState.status === "failed") {
                setGenerationWorkspaceText(
                    outputMeta,
                    uiT("generation.loading.output_failed")
                );
            }
        };

        const renderGenerationWorkspaceLoading = () => {
            const state = generationWorkspaceLoadingState;
            const visible = state.status !== "idle";
            const busy = generationWorkspaceIsBusy(state.status);
            const progress = generationWorkspaceSafeProgress(state.progress);
            const outputTabs = document.getElementById("output-tabs");

            installGenerationWorkspaceTabIndicators();
            installGenerationActionSpinner();

            if (outputTabs) {
                outputTabs.dataset.generationState = state.status;
                outputTabs.setAttribute("aria-busy", busy ? "true" : "false");
            }

            Object.entries(generationWorkspaceLoadingHosts).forEach(([kind, id]) => {
                const host = document.getElementById(id);
                const shell = host?.querySelector(".generation-workspace-loading");
                if (!host || !shell) return;
                host.classList.toggle("is-visible", visible);
                host.setAttribute("aria-hidden", visible ? "false" : "true");
                shell.dataset.state = state.status;
                shell.setAttribute("aria-hidden", visible ? "false" : "true");
                shell.setAttribute("role", state.status === "failed" ? "alert" : "status");

                const phase = shell.querySelector("[data-generation-loading-phase]");
                const title = shell.querySelector("[data-generation-loading-title]");
                const message = shell.querySelector("[data-generation-loading-message]");
                const percent = shell.querySelector("[data-generation-loading-percent]");
                const bar = shell.querySelector("[data-generation-loading-bar]");
                const track = shell.querySelector('[role="progressbar"]');
                const hint = shell.querySelector(".generation-loading-console-hint");
                setGenerationWorkspaceText(phase, generationWorkspacePhase(state));
                setGenerationWorkspaceText(
                    title,
                    generationWorkspaceDisplayTitle(kind, state)
                );
                setGenerationWorkspaceText(
                    message,
                    generationWorkspaceDisplayMessage(kind, state)
                );
                setGenerationWorkspaceText(percent, Math.round(progress) + "%");
                if (bar) bar.style.width = progress + "%";
                if (track) {
                    track.setAttribute("aria-valuenow", String(Math.round(progress)));
                    track.setAttribute("aria-valuetext", Math.round(progress) + "%");
                    track.setAttribute("aria-label", uiT("generation.loading.progress_label"));
                }
                setGenerationWorkspaceText(
                    hint,
                    uiT("generation.loading.console_hint")
                );
            });

            generationWorkspaceSurfaceIds.forEach((id) => {
                const surface = document.getElementById(id);
                if (!surface) return;
                surface.setAttribute("aria-busy", busy ? "true" : "false");
                surface.toggleAttribute("inert", visible);
            });

            generationWorkspaceTabIds.forEach((tabId) => {
                const button = outputTabs?.querySelector(
                    'button[role="tab"][data-tab-id="' + tabId + '"]'
                );
                if (button) button.setAttribute("aria-busy", busy ? "true" : "false");
            });

            const generateButton = document.getElementById("generate-3d-button");
            generateButton?.classList.toggle("is-generation-running", busy);
            generateButton?.setAttribute("aria-busy", busy ? "true" : "false");
            syncGenerationSidebarLoading(busy);
        };

        const hideGenerationWorkspaceLoading = (uid) => {
            if (uid && generationWorkspaceLoadingState.uid !== uid) return;
            clearGenerationWorkspaceReadyWait();
            clearGenerationWorkspaceCompleteTimer();
            generationWorkspaceLoadingState = {
                ...generationWorkspaceLoadingState,
                status: "idle",
                progress: 100,
                message: "",
            };
            renderGenerationWorkspaceLoading();
        };

        const completeGenerationWorkspaceLoading = (uid) => {
            if (
                generationWorkspaceLoadingState.uid !== uid
                || generationWorkspaceLoadingState.status !== "finalizing"
            ) return;
            clearGenerationWorkspaceReadyWait();
            clearGenerationWorkspaceCompleteTimer();
            generationWorkspaceLoadingState = {
                ...generationWorkspaceLoadingState,
                status: "complete",
                progress: 100,
                message: uiT("generation.loading.complete_body"),
            };
            if (
                generationConsoleLastManifest?.generation_uid === uid
                && typeof updateGenerationDetails === "function"
            ) {
                updateGenerationDetails(generationConsoleLastManifest);
            }
            renderGenerationWorkspaceLoading();
            generationWorkspaceCompleteTimer = window.setTimeout(
                () => hideGenerationWorkspaceLoading(uid),
                700
            );
        };

        const generationWorkspacePathMatchesUid = (pathname, uid) => (
            pathname === "/generation-viewer/" + uid
            || pathname.startsWith("/static/" + uid + "/")
        );

        const generationWorkspaceFrameTargetsUid = (frame, uid) => {
            const source = String(frame?.getAttribute("src") || "");
            if (!source || !uid) return false;
            try {
                const pathname = decodeURIComponent(
                    new URL(source, window.location.href).pathname
                );
                return generationWorkspacePathMatchesUid(pathname, uid);
            } catch {
                return false;
            }
        };

        const generationWorkspaceFrameLoadedForUid = (frame, uid) => {
            if (!generationWorkspaceFrameTargetsUid(frame, uid)) return false;
            try {
                const pathname = decodeURIComponent(
                    frame.contentWindow?.location?.pathname || ""
                );
                return generationWorkspacePathMatchesUid(pathname, uid)
                    && frame.contentDocument?.readyState === "complete";
            } catch {
                return false;
            }
        };

        const generationWorkspaceResultFrame = (uid) => {
            if (!uid) return null;
            return Array.from(document.querySelectorAll("#mesh-viewer iframe")).find(
                (frame) => generationWorkspaceFrameTargetsUid(frame, uid)
            ) || null;
        };

        const waitForGenerationWorkspaceResult = (uid) => {
            clearGenerationWorkspaceReadyWait();
            let settlingFrame = null;
            const settleFrame = (frame) => {
                if (
                    settlingFrame === frame
                    || generationWorkspaceLoadingState.uid !== uid
                ) return;
                settlingFrame = frame;
                const ready = () => {
                    if (
                        generationWorkspaceResultFrame(uid) !== frame
                        || !generationWorkspaceFrameLoadedForUid(frame, uid)
                    ) return;
                    window.setTimeout(
                        () => completeGenerationWorkspaceLoading(uid),
                        180
                    );
                };
                frame.addEventListener("load", ready, {once: true});
                ready();
            };
            const check = () => {
                const frame = generationWorkspaceResultFrame(uid);
                if (frame) settleFrame(frame);
            };
            const outputTabs = document.getElementById("output-tabs");
            if (outputTabs) {
                generationWorkspaceReadyObserver = new MutationObserver(check);
                generationWorkspaceReadyObserver.observe(outputTabs, {
                    attributes: true,
                    attributeFilter: ["src"],
                    childList: true,
                    subtree: true,
                });
            }
            generationWorkspaceReadyTimer = window.setInterval(check, 500);
            check();
        };

        const startGenerationWorkspaceLoading = (uid, resumed = false) => {
            if (!uid) return;
            clearGenerationWorkspaceReadyWait();
            clearGenerationWorkspaceCompleteTimer();
            generationWorkspaceLoadingState = {
                status: "queued",
                uid,
                progress: 1,
                message: resumed
                    ? uiT("generation.console.progress.restoring")
                    : uiT("generation.console.progress.dispatching"),
                resumed,
            };
            renderGenerationWorkspaceLoading();
        };

        const syncGenerationWorkspaceProgress = (progress, message = "") => {
            if (!generationWorkspaceLoadingState.uid) return;
            if (
                generationWorkspaceLoadingState.status !== "queued"
                && generationWorkspaceLoadingState.status !== "running"
            ) return;
            const safeProgress = generationWorkspaceSafeProgress(progress);
            generationWorkspaceLoadingState = {
                ...generationWorkspaceLoadingState,
                status: safeProgress > 2 ? "running" : "queued",
                progress: safeProgress,
                message: message || generationWorkspaceLoadingState.message,
            };
            renderGenerationWorkspaceLoading();
        };

        const syncGenerationWorkspaceManifest = (manifest, message = "") => {
            const uid = manifest?.generation_uid;
            if (!uid || uid !== generationWorkspaceLoadingState.uid) return;
            const status = String(manifest.status || "processing");
            if (status === "failed") {
                clearGenerationWorkspaceReadyWait();
                clearGenerationWorkspaceCompleteTimer();
                generationWorkspaceLoadingState = {
                    ...generationWorkspaceLoadingState,
                    status: "failed",
                    progress: generationWorkspaceSafeProgress(manifest.progress),
                    message: uiT("generation.loading.failed_body"),
                };
                renderGenerationWorkspaceLoading();
                return;
            }
            if (status === "completed") {
                if (
                    generationWorkspaceLoadingState.status === "idle"
                    && generationWorkspaceResultFrame(uid)
                ) return;
                generationWorkspaceLoadingState = {
                    ...generationWorkspaceLoadingState,
                    status: "finalizing",
                    progress: 100,
                    message: uiT("generation.loading.finalizing_body"),
                };
                renderGenerationWorkspaceLoading();
                waitForGenerationWorkspaceResult(uid);
                return;
            }
            generationWorkspaceLoadingState = {
                ...generationWorkspaceLoadingState,
                status: "running",
                progress: generationWorkspaceSafeProgress(manifest.progress),
                message: message || generationWorkspaceLoadingState.message,
            };
            renderGenerationWorkspaceLoading();
        };

        const installGenerationWorkspaceLoading = () => {
            renderGenerationWorkspaceLoading();
        };

        window.addEventListener("ui-language-change", () => {
            renderGenerationWorkspaceLoading();
        });
