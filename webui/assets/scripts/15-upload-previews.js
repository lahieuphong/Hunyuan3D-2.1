
        const stableUploadPreviewIds = [
            "mv-image-front",
            "mv-image-back",
            "mv-image-left",
            "mv-image-right",
            "ten-image-front",
            "ten-image-front_right",
            "ten-image-right",
            "ten-image-back_right",
            "ten-image-back",
            "ten-image-back_left",
            "ten-image-left",
            "ten-image-front_left",
            "ten-image-high_front",
            "ten-image-high_back",
        ];
        const stableUploadPreviewRetryDelays = [250, 700, 1500];
        const stableUploadPreviewStates = new Map();
        const stableUploadDeadlineMs = 300000;
        const stableNativePreviewDeadlineMs = 5000;
        const stableUploadPreviewFadeMs = 180;
        const uploadActionTranslationKeys = Object.freeze({
            capture: "upload.capture_camera",
            paste: "upload.paste_clipboard",
            remove: "upload.remove_image",
            upload: "upload.upload_file",
        });

        const uploadActionButtonMatches = (button, action) => {
            const key = uploadActionTranslationKeys[action];
            const label = button?.getAttribute("aria-label");
            return Boolean(
                key
                && label
                && (
                    label === uiT(key, {}, "en")
                    || label === uiT(key, {}, "zh-CN")
                )
            );
        };

        const uploadActionButton = (root, action) => (
            Array.from(root.querySelectorAll("button[aria-label]")).find(
                (button) => uploadActionButtonMatches(button, action)
            ) ?? null
        );

        const closestUploadActionButton = (target, action) => {
            const button = target.closest?.("button[aria-label]");
            return uploadActionButtonMatches(button, action) ? button : null;
        };

        const uploadPreviewHost = (tile) => (
            tile.querySelector('[data-testid="image"] .upload-container')
        );

        const uploadFileLooksLikeImage = (file) => (
            file instanceof File
            && (
                file.type.startsWith("image/")
                || /\.(?:avif|bmp|gif|jpe?g|png|svg|webp)$/i.test(file.name)
            )
        );

        const nativePreviewIsReady = (image) => (
            image.complete && image.naturalWidth > 0 && image.naturalHeight > 0
        );

        const nativePreviewBelongsToRun = (state, image) => {
            const source = image.currentSrc || image.src || "";
            return (
                image !== state.previousNativeImage
                || source !== state.previousNativeSource
                || state.sawNativeAbsent
            );
        };

        const refreshNativeUploadStatus = (tile) => {
            tile.querySelectorAll(
                ":scope > .wrap.default.full.mv-native-upload-status-suppressed, "
                + ":scope > .wrap.default.full"
                + '[data-mv-native-upload-status-suppressed="true"]'
            ).forEach((status) => {
                if (
                    status.dataset.mvNativeUploadStatusSuppressed !== "true"
                    || status.querySelector(".error")
                ) return;
                status.classList.remove("mv-native-upload-status-suppressed");
                delete status.dataset.mvNativeUploadStatusSuppressed;
                const previousAriaHidden = status.dataset
                    .mvNativeUploadStatusPreviousAriaHidden;
                delete status.dataset.mvNativeUploadStatusPreviousAriaHidden;
                if (previousAriaHidden === "__absent__") {
                    status.removeAttribute("aria-hidden");
                } else if (previousAriaHidden !== undefined) {
                    status.setAttribute("aria-hidden", previousAriaHidden);
                }
            });
        };

        const suppressStaleNativeUploadStatus = (tile) => {
            refreshNativeUploadStatus(tile);
            tile.querySelectorAll(":scope > .wrap.default.full").forEach((status) => {
                if (!status.querySelector(".error")) return;
                if (status.dataset.mvNativeUploadStatusSuppressed !== "true") {
                    status.dataset.mvNativeUploadStatusPreviousAriaHidden = (
                        status.hasAttribute("aria-hidden")
                            ? status.getAttribute("aria-hidden")
                            : "__absent__"
                    );
                    status.dataset.mvNativeUploadStatusSuppressed = "true";
                }
                if (!status.classList.contains("mv-native-upload-status-suppressed")) {
                    status.classList.add("mv-native-upload-status-suppressed");
                }
                if (status.getAttribute("aria-hidden") !== "true") {
                    status.setAttribute("aria-hidden", "true");
                }
            });
        };

        const releaseUploadPreviewUrl = (state) => {
            if (!state.objectUrl) return;
            URL.revokeObjectURL(state.objectUrl);
            state.objectUrl = null;
        };

        const removeUploadPreviewNodes = (state) => {
            state.preview?.remove();
            state.status?.remove();
            state.preview = null;
            state.status = null;
        };

        const ensureUploadPreviewMounted = (state) => {
            if (!state.preview) return false;
            const host = uploadPreviewHost(state.tile);
            if (!host) return false;
            if (!state.preview.isConnected || state.preview.parentElement !== host) {
                host.append(state.preview);
            }
            if (
                state.status
                && (!state.status.isConnected || state.status.parentElement !== host)
            ) host.append(state.status);
            return state.preview.isConnected && state.preview.parentElement === host;
        };

        const cancelUploadPreviewTimers = (state) => {
            window.clearTimeout(state.reconcileTimer);
            window.clearTimeout(state.retryTimer);
            window.clearTimeout(state.settleTimer);
            state.reconcileTimer = 0;
            state.retryTimer = 0;
            state.settleTimer = 0;
        };

        const resetStableUploadPreview = (state) => {
            state.runId += 1;
            cancelUploadPreviewTimers(state);
            state.nativeListenerController?.abort();
            state.nativeListenerController = null;
            state.nativeListenerImage = null;
            state.acceptedAt = 0;
            state.deadlineAt = 0;
            state.mode = "idle";
            state.previousNativeError = null;
            state.previousNativeErrorText = "";
            state.previousNativeImage = null;
            state.previousNativeSource = "";
            state.previousRemoveButton = null;
            state.retryCount = 0;
            state.sawNativeErrorAbsent = false;
            state.sawNativeAbsent = false;
            state.sawRemoveAbsent = false;
            state.sawUploadProgress = false;
            state.startedAt = 0;
            state.tile.classList.remove(
                "is-uploading-preview",
                "is-upload-preview-error",
                "is-upload-preview-local",
                "is-upload-preview-settling"
            );
            removeUploadPreviewNodes(state);
            releaseUploadPreviewUrl(state);
        };

        const disposeStableUploadPreview = (state) => {
            resetStableUploadPreview(state);
            state.observer?.disconnect();
            state.observer = null;
            state.listenerController?.abort();
            state.listenerController = null;
            delete state.tile.dataset.stableUploadPreviewWired;
        };

        const ensureUploadPreviewStatus = (state, message) => {
            const host = uploadPreviewHost(state.tile);
            if (!host) return null;
            if (!state.status) {
                state.status = document.createElement("span");
                state.status.className = "mv-upload-preview-status";
                state.status.setAttribute("role", "status");
                state.status.setAttribute("aria-live", "polite");
            }
            if (!state.status.isConnected || state.status.parentElement !== host) {
                host.append(state.status);
            }
            if (state.status.textContent !== message) {
                state.status.textContent = message;
            }
            return state.status;
        };

        const failStableUploadPreview = (state, runId, message) => {
            if (state.runId !== runId) return;
            cancelUploadPreviewTimers(state);
            state.mode = "error";
            state.startedAt = 0;
            state.retryCount = 0;
            state.tile.classList.remove(
                "is-uploading-preview",
                "is-upload-preview-local"
            );
            state.tile.classList.add("is-upload-preview-error");
            state.preview?.remove();
            state.preview = null;
            releaseUploadPreviewUrl(state);
            suppressStaleNativeUploadStatus(state.tile);
            ensureUploadPreviewStatus(state, message);
        };

        const settleStableLocalPreview = (state, runId) => {
            if (state.runId !== runId) return;
            if (!ensureUploadPreviewMounted(state)) {
                failStableUploadPreview(
                    state,
                    runId,
                    uiT("upload.preview_failed")
                );
                return;
            }
            cancelUploadPreviewTimers(state);
            state.mode = "fallback";
            state.startedAt = 0;
            state.tile.classList.remove(
                "is-uploading-preview",
                "is-upload-preview-error"
            );
            state.tile.classList.add("is-upload-preview-local");
            state.status?.remove();
            state.status = null;
        };

        const finishStableUploadPreview = (state, runId) => {
            if (state.runId !== runId || state.mode === "settling") return;
            const nativeImage = state.nativeListenerImage;
            if (!nativeImage || !nativePreviewIsReady(nativeImage)) return;
            state.mode = "settling";
            cancelUploadPreviewTimers(state);
            suppressStaleNativeUploadStatus(state.tile);
            state.status?.remove();
            state.status = null;
            void (async () => {
                if (typeof nativeImage.decode === "function") {
                    try {
                        await nativeImage.decode();
                    } catch (_error) {
                        // The load event already confirmed a usable image.
                    }
                }
                await new Promise((resolve) => {
                    window.requestAnimationFrame(() => {
                        window.requestAnimationFrame(resolve);
                    });
                });
                if (
                    state.runId !== runId
                    || state.nativeListenerImage !== nativeImage
                    || !nativePreviewIsReady(nativeImage)
                ) return;
                state.tile.classList.add("is-upload-preview-settling");
                const reduceMotion = window.matchMedia(
                    "(prefers-reduced-motion: reduce)"
                ).matches;
                state.settleTimer = window.setTimeout(() => {
                    state.settleTimer = 0;
                    if (state.runId === runId) resetStableUploadPreview(state);
                }, reduceMotion ? 0 : stableUploadPreviewFadeMs);
            })();
        };

        const scheduleUploadPreviewReconcile = (state, runId, delay = 0) => {
            if (state.runId !== runId) return;
            window.clearTimeout(state.reconcileTimer);
            state.reconcileTimer = window.setTimeout(() => {
                state.reconcileTimer = 0;
                if (state.runId === runId) reconcileStableUploadPreview(state, runId);
            }, delay);
        };

        const retryNativeUploadPreview = (state, runId, image) => {
            if (
                state.runId !== runId
                || !nativePreviewBelongsToRun(state, image)
            ) return;
            if (!image.isConnected) {
                scheduleUploadPreviewReconcile(state, runId, 100);
                return;
            }
            if (nativePreviewIsReady(image)) {
                finishStableUploadPreview(state, runId);
                return;
            }
            if (state.retryTimer) return;

            const retryIndex = state.retryCount;
            if (retryIndex >= stableUploadPreviewRetryDelays.length) {
                if (
                    state.preview
                    && uploadActionButton(state.tile, "remove")
                ) {
                    settleStableLocalPreview(state, runId);
                    return;
                }
                failStableUploadPreview(
                    state,
                    runId,
                    uiT("upload.preview_failed")
                );
                return;
            }

            state.retryCount += 1;
            state.retryTimer = window.setTimeout(() => {
                state.retryTimer = 0;
                if (state.runId !== runId) return;
                if (!image.isConnected) {
                    scheduleUploadPreviewReconcile(state, runId, 100);
                    return;
                }
                if (nativePreviewIsReady(image)) {
                    finishStableUploadPreview(state, runId);
                    return;
                }
                const source = image.dataset.stablePreviewSource
                    || image.currentSrc
                    || image.src;
                if (!source || source.startsWith("data:") || source.startsWith("blob:")) {
                    retryNativeUploadPreview(state, runId, image);
                    return;
                }
                image.dataset.stablePreviewSource = source;
                const retryUrl = new URL(source, window.location.href);
                retryUrl.searchParams.set("_preview_retry", String(state.retryCount));
                image.src = retryUrl.href;
            }, stableUploadPreviewRetryDelays[retryIndex]);
        };

        const wireNativeUploadPreview = (state, runId, image) => {
            const wiredRun = String(runId);
            if (
                image.dataset.stablePreviewRun === wiredRun
                && state.nativeListenerImage === image
            ) return;
            state.nativeListenerController?.abort();
            state.nativeListenerController = new AbortController();
            state.nativeListenerImage = image;
            const listenerOptions = {
                signal: state.nativeListenerController.signal,
            };
            image.dataset.stablePreviewRun = wiredRun;
            image.dataset.stablePreviewSource = image.currentSrc || image.src || "";
            image.addEventListener("load", () => {
                if (
                    state.runId === runId
                    && nativePreviewBelongsToRun(state, image)
                    && nativePreviewIsReady(image)
                ) finishStableUploadPreview(state, runId);
            }, listenerOptions);
            image.addEventListener("error", () => {
                retryNativeUploadPreview(state, runId, image);
            }, listenerOptions);
        };

        const reconcileStableUploadPreview = (state, runId) => {
            if (state.runId !== runId) return;
            if (!state.tile.isConnected) {
                disposeStableUploadPreview(state);
                stableUploadPreviewStates.delete(state.id);
                return;
            }

            refreshNativeUploadStatus(state.tile);
            if (state.mode === "uploading" || state.mode === "fallback") {
                ensureUploadPreviewMounted(state);
            }
            suppressStaleNativeUploadStatus(state.tile);
            const nativeImage = state.tile.querySelector(".image-frame img");
            const nativeIsCurrent = Boolean(
                nativeImage && nativePreviewBelongsToRun(state, nativeImage)
            );
            const nativeError = state.tile.querySelector(
                ":scope > .wrap.default.full .error"
            );
            if (!nativeError) state.sawNativeErrorAbsent = true;
            const nativeErrorIsCurrent = Boolean(
                nativeError
                && (
                    nativeError !== state.previousNativeError
                    || nativeError.textContent !== state.previousNativeErrorText
                    || state.sawNativeErrorAbsent
                )
            );

            if (state.mode === "fallback") {
                if (nativeImage && nativeIsCurrent) {
                    wireNativeUploadPreview(state, runId, nativeImage);
                    if (nativePreviewIsReady(nativeImage)) {
                        finishStableUploadPreview(state, runId);
                    } else if (nativeImage.complete) {
                        retryNativeUploadPreview(state, runId, nativeImage);
                    }
                } else if (
                    !nativeImage
                    && !uploadActionButton(state.tile, "remove")
                ) {
                    resetStableUploadPreview(state);
                }
                return;
            }
            if (state.mode !== "uploading") return;

            if (nativeErrorIsCurrent) {
                failStableUploadPreview(
                    state,
                    runId,
                    uiT("upload.failed")
                );
                return;
            }

            if (nativeImage && nativeIsCurrent) {
                wireNativeUploadPreview(state, runId, nativeImage);
                if (nativePreviewIsReady(nativeImage)) {
                    finishStableUploadPreview(state, runId);
                    return;
                }
                if (nativeImage.complete) {
                    retryNativeUploadPreview(state, runId, nativeImage);
                }
            }

            const progress = state.tile.querySelector(
                ".upload-container .wrap.progress, .upload-container .uploading"
            );
            if (progress) {
                state.sawUploadProgress = true;
                if (!nativeImage) state.sawNativeAbsent = true;
                if (Date.now() >= state.deadlineAt) {
                    failStableUploadPreview(
                        state,
                        runId,
                        uiT("upload.timed_out")
                    );
                    return;
                }
                ensureUploadPreviewStatus(state, uiT("upload.uploading_preview"));
                scheduleUploadPreviewReconcile(state, runId, 400);
                return;
            }

            if (!nativeImage) state.sawNativeAbsent = true;
            const accepted = uploadActionButton(state.tile, "remove");
            if (!accepted) state.sawRemoveAbsent = true;
            const acceptedBelongsToRun = Boolean(
                accepted
                && (
                    nativeIsCurrent
                    || (
                        state.sawUploadProgress
                        && state.sawRemoveAbsent
                        && accepted !== state.previousRemoveButton
                    )
                )
            );
            if (acceptedBelongsToRun) {
                if (!state.acceptedAt) state.acceptedAt = Date.now();
                if (Date.now() - state.acceptedAt >= stableNativePreviewDeadlineMs) {
                    if (ensureUploadPreviewMounted(state)) {
                        settleStableLocalPreview(state, runId);
                    } else {
                        failStableUploadPreview(
                            state,
                            runId,
                            uiT("upload.preview_failed")
                        );
                    }
                    return;
                }
                ensureUploadPreviewStatus(state, uiT("upload.preparing_preview"));
                scheduleUploadPreviewReconcile(state, runId, 250);
                return;
            }

            if (Date.now() - state.startedAt < 1500) {
                scheduleUploadPreviewReconcile(state, runId, 150);
                return;
            }
            failStableUploadPreview(
                state,
                runId,
                uiT("upload.interrupted")
            );
        };

        const startStableUploadPreview = (state, file) => {
            resetStableUploadPreview(state);
            if (!uploadFileLooksLikeImage(file)) return;

            const host = uploadPreviewHost(state.tile);
            if (!host) return;
            const previousNative = state.tile.querySelector(".image-frame img");
            const previousNativeError = state.tile.querySelector(
                ":scope > .wrap.default.full .error"
            );
            state.previousNativeImage = previousNative;
            state.previousNativeSource = previousNative?.currentSrc
                || previousNative?.src
                || "";
            state.previousNativeError = previousNativeError;
            state.previousNativeErrorText = previousNativeError?.textContent || "";
            state.previousRemoveButton = uploadActionButton(state.tile, "remove");
            state.sawNativeErrorAbsent = !previousNativeError;
            state.sawRemoveAbsent = !state.previousRemoveButton;
            state.startedAt = Date.now();
            state.deadlineAt = state.startedAt + stableUploadDeadlineMs;
            state.mode = "uploading";
            const runId = state.runId;
            state.objectUrl = URL.createObjectURL(file);
            state.preview = document.createElement("img");
            state.preview.className = "mv-upload-local-preview";
            state.preview.alt = "";
            state.preview.decoding = "async";
            state.preview.src = state.objectUrl;
            state.preview.addEventListener("error", () => {
                failStableUploadPreview(
                    state,
                    runId,
                    uiT("upload.cannot_preview")
                );
            }, {once: true});
            host.append(state.preview);
            state.tile.classList.add("is-uploading-preview");
            suppressStaleNativeUploadStatus(state.tile);
            ensureUploadPreviewStatus(state, uiT("upload.uploading_preview"));
            scheduleUploadPreviewReconcile(state, runId, 100);
        };

        const wireStableUploadPreview = (id, tile) => {
            const state = {
                acceptedAt: 0,
                deadlineAt: 0,
                id,
                listenerController: new AbortController(),
                mode: "idle",
                nativeListenerController: null,
                nativeListenerImage: null,
                objectUrl: null,
                observer: null,
                preview: null,
                previousNativeError: null,
                previousNativeErrorText: "",
                previousNativeImage: null,
                previousNativeSource: "",
                previousRemoveButton: null,
                reconcileTimer: 0,
                retryCount: 0,
                retryTimer: 0,
                runId: 0,
                sawNativeErrorAbsent: false,
                sawNativeAbsent: false,
                sawRemoveAbsent: false,
                sawUploadProgress: false,
                settleTimer: 0,
                startedAt: 0,
                status: null,
                tile,
            };
            tile.dataset.stableUploadPreviewWired = "true";
            stableUploadPreviewStates.set(id, state);

            const listenerOptions = {
                capture: true,
                signal: state.listenerController.signal,
            };
            tile.addEventListener("change", (event) => {
                const input = event.target.closest?.(
                    'input[type="file"][data-testid="file-upload"]'
                );
                if (!input || !tile.contains(input)) return;
                startStableUploadPreview(state, input.files?.[0]);
            }, listenerOptions);
            tile.addEventListener("drop", (event) => {
                const file = event.dataTransfer?.files?.[0];
                if (file) startStableUploadPreview(state, file);
            }, listenerOptions);
            tile.addEventListener("click", (event) => {
                const uploadSource = closestUploadActionButton(
                    event.target,
                    "upload"
                );
                if (
                    uploadSource
                    && tile.contains(uploadSource)
                    && uploadSource.classList.contains("selected")
                ) {
                    const input = tile.querySelector(
                        'input[type="file"][data-testid="file-upload"]'
                    );
                    if (!input) return;
                    event.preventDefault();
                    event.stopPropagation();
                    event.stopImmediatePropagation();
                    input.value = "";
                    input.click();
                    return;
                }

                const remove = closestUploadActionButton(event.target, "remove");
                const switchesSource = (
                    closestUploadActionButton(event.target, "capture")
                    || closestUploadActionButton(event.target, "paste")
                );
                if (
                    (remove && tile.contains(remove))
                    || (switchesSource && tile.contains(switchesSource))
                ) resetStableUploadPreview(state);
            }, listenerOptions);

            state.observer = new MutationObserver(() => {
                refreshNativeUploadStatus(state.tile);
                if (state.mode === "uploading" || state.mode === "fallback") {
                    scheduleUploadPreviewReconcile(state, state.runId);
                } else if (state.mode === "error") {
                    const nativeImage = state.tile.querySelector(".image-frame img");
                    if (
                        nativeImage
                        && nativePreviewBelongsToRun(state, nativeImage)
                        && uploadActionButton(state.tile, "remove")
                    ) {
                        wireNativeUploadPreview(state, state.runId, nativeImage);
                        if (nativePreviewIsReady(nativeImage)) {
                            finishStableUploadPreview(state, state.runId);
                            return;
                        }
                    }
                    suppressStaleNativeUploadStatus(state.tile);
                }
            });
            state.observer.observe(tile, {
                attributeFilter: ["class", "src"],
                attributes: true,
                childList: true,
                subtree: true,
            });
        };

        const installStableUploadPreviews = () => {
            stableUploadPreviewIds.forEach((id) => {
                const tile = document.getElementById(id);
                const existing = stableUploadPreviewStates.get(id);
                if (existing && existing.tile !== tile) {
                    disposeStableUploadPreview(existing);
                    stableUploadPreviewStates.delete(id);
                }
                if (tile) refreshNativeUploadStatus(tile);
                if (tile && stableUploadPreviewStates.get(id)?.tile !== tile) {
                    wireStableUploadPreview(id, tile);
                }
            });
        };
