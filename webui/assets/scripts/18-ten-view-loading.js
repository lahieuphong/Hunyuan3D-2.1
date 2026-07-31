
        const tenViewHistoryPreviewState = {
            hydrationExpired: false,
            hydrationTimeoutId: 0,
            phase: "idle",
            root: null,
            routeKey: "",
            slots: new Map(),
        };
        const tenViewHistoryExpectedImageCount = 10;
        const tenViewHistoryLoadTimeoutMs = 12000;
        const tenViewHistoryPriorityImageCount = 4;

        const tenViewHistoryRouteContext = () => {
            const url = new URL(window.location.href);
            const generation = url.searchParams.get("generation")?.trim() || "";
            if (!generation) return null;
            const reviewState = document.querySelector(
                '#history-review-state [data-history-review-active="true"]'
            );
            const inputMode = reviewState
                ? reviewState.dataset.inputMode
                : url.searchParams.get("tab") === "ten-view"
                    ? "ten"
                    : "";
            if (inputMode !== "ten") return null;
            return {
                generation,
                key: `ten:${generation}`,
            };
        };

        const tenViewHistoryImageIsReady = (image) => (
            image.complete
            && image.naturalWidth > 0
            && image.naturalHeight > 0
        );

        const tenViewHistoryImageSource = (image) => (
            image.currentSrc || image.src || ""
        );

        const configureTenViewHistoryImage = (
            image,
            index
        ) => {
            image.decoding = "async";
            image.loading = "eager";
            const priority = index < tenViewHistoryPriorityImageCount
                ? "high"
                : "low";
            image.fetchPriority = priority;
            image.setAttribute("fetchpriority", priority);
            return tenViewHistoryImageSource(image);
        };

        const setTenViewHistoryTileState = (record, tile, status) => {
            record.status = status;
            record.tile = tile;
            tile.dataset.tenViewPreviewState = status;
        };

        const syncTenViewHistoryLoadingPhase = () => {
            const {root} = tenViewHistoryPreviewState;
            if (!root?.isConnected) return;
            const tiles = [...root.querySelectorAll(".ten-view-image")];
            const hasPendingTile = tiles.some((tile) => (
                tile.dataset.tenViewPreviewState === "loading"
            ));
            const isHydrating = (
                tiles.length !== tenViewHistoryExpectedImageCount
                && !tenViewHistoryPreviewState.hydrationExpired
            );
            const isBusy = hasPendingTile || isHydrating;
            tenViewHistoryPreviewState.phase = isBusy ? "loading" : "ready";
            root.dataset.tenViewLoadingState = isBusy ? "loading" : "ready";
            if (isBusy) {
                root.setAttribute("aria-busy", "true");
            } else {
                root.removeAttribute("aria-busy");
            }
        };

        const stopTenViewHistorySlot = (record) => {
            record.controller?.abort();
            record.controller = null;
            window.clearTimeout(record.loadTimeoutId);
            record.loadTimeoutId = 0;
        };

        const resetTenViewHistoryLoading = () => {
            window.clearTimeout(tenViewHistoryPreviewState.hydrationTimeoutId);
            tenViewHistoryPreviewState.hydrationTimeoutId = 0;
            tenViewHistoryPreviewState.slots.forEach((record) => {
                stopTenViewHistorySlot(record);
                record.tile?.removeAttribute("data-ten-view-preview-state");
            });
            tenViewHistoryPreviewState.root?.removeAttribute(
                "data-ten-view-loading-state"
            );
            tenViewHistoryPreviewState.root?.removeAttribute("aria-busy");
            tenViewHistoryPreviewState.hydrationExpired = false;
            tenViewHistoryPreviewState.phase = "idle";
            tenViewHistoryPreviewState.root = null;
            tenViewHistoryPreviewState.routeKey = "";
            tenViewHistoryPreviewState.slots.clear();
        };

        const finishTenViewHistorySlot = (record, tile, status) => {
            if (record.tile !== tile) return;
            window.clearTimeout(record.loadTimeoutId);
            record.loadTimeoutId = 0;
            setTenViewHistoryTileState(record, tile, status);
            syncTenViewHistoryLoadingPhase();
        };

        const revealTenViewHistoryImage = async (
            record,
            tile,
            image,
            source,
            runId
        ) => {
            if (!tenViewHistoryImageIsReady(image)) {
                finishTenViewHistorySlot(record, tile, "error");
                return;
            }
            if (typeof image.decode === "function") {
                try {
                    await image.decode();
                } catch (_error) {
                    // A loaded image is still usable if decode() races a
                    // Gradio node replacement or is unsupported.
                }
            }
            await new Promise((resolve) => {
                window.requestAnimationFrame(() => {
                    window.requestAnimationFrame(resolve);
                });
            });
            if (
                record.runId !== runId
                || record.image !== image
                || record.source !== source
                || !tenViewHistoryImageIsReady(image)
            ) return;
            record.controller?.abort();
            record.controller = null;
            finishTenViewHistorySlot(record, tile, "ready");
        };

        const beginTenViewHistorySlotLoading = (
            record,
            tile,
            image,
            source
        ) => {
            stopTenViewHistorySlot(record);
            record.runId += 1;
            const runId = record.runId;
            const controller = new AbortController();
            record.controller = controller;
            record.image = image;
            record.source = source;
            setTenViewHistoryTileState(record, tile, "loading");

            const reveal = () => {
                void revealTenViewHistoryImage(
                    record,
                    tile,
                    image,
                    source,
                    runId
                );
            };
            const fail = () => {
                if (
                    record.runId !== runId
                    || record.image !== image
                    || record.source !== source
                ) return;
                record.controller?.abort();
                record.controller = null;
                finishTenViewHistorySlot(record, tile, "error");
            };
            image.addEventListener("load", reveal, {
                once: true,
                signal: controller.signal,
            });
            image.addEventListener("error", fail, {
                once: true,
                signal: controller.signal,
            });
            record.loadTimeoutId = window.setTimeout(() => {
                if (
                    record.runId === runId
                    && record.image === image
                    && record.source === source
                ) finishTenViewHistorySlot(record, tile, "error");
            }, tenViewHistoryLoadTimeoutMs);

            if (tenViewHistoryImageIsReady(image)) {
                reveal();
            } else if (image.complete) {
                fail();
            }
        };

        const markTenViewHistorySlotWaiting = (record, tile) => {
            if (record.tile !== tile || record.status !== "loading") {
                stopTenViewHistorySlot(record);
                record.image = null;
                setTenViewHistoryTileState(record, tile, "loading");
            }
        };

        const startTenViewHistoryRoute = (context, root) => {
            resetTenViewHistoryLoading();
            tenViewHistoryPreviewState.root = root;
            tenViewHistoryPreviewState.routeKey = context.key;
            tenViewHistoryPreviewState.phase = "loading";
            root.dataset.tenViewLoadingState = "loading";
            root.setAttribute("aria-busy", "true");
            tenViewHistoryPreviewState.hydrationTimeoutId = window.setTimeout(
                () => {
                    tenViewHistoryPreviewState.hydrationExpired = true;
                    tenViewHistoryPreviewState.slots.forEach((record) => {
                        if (
                            record.status === "loading"
                            && !record.image
                            && record.tile
                        ) {
                            finishTenViewHistorySlot(
                                record,
                                record.tile,
                                "error"
                            );
                        }
                    });
                    syncTenViewHistoryLoadingPhase();
                },
                tenViewHistoryLoadTimeoutMs
            );
        };

        const installTenViewHistoryLoading = () => {
            const context = tenViewHistoryRouteContext();
            const root = document.getElementById("prompt-mode-tabs");
            if (!context || !root) {
                if (tenViewHistoryPreviewState.phase !== "idle") {
                    resetTenViewHistoryLoading();
                }
                return;
            }
            if (context.key !== tenViewHistoryPreviewState.routeKey) {
                startTenViewHistoryRoute(context, root);
            } else if (tenViewHistoryPreviewState.root !== root) {
                tenViewHistoryPreviewState.root?.removeAttribute(
                    "data-ten-view-loading-state"
                );
                tenViewHistoryPreviewState.root?.removeAttribute("aria-busy");
                tenViewHistoryPreviewState.root = root;
            }

            const tiles = [...root.querySelectorAll(".ten-view-image")];
            tiles.forEach((tile, index) => {
                const slotId = tile.id || `ten-view-slot-${index}`;
                let record = tenViewHistoryPreviewState.slots.get(slotId);
                if (!record) {
                    record = {
                        controller: null,
                        image: null,
                        loadTimeoutId: 0,
                        runId: 0,
                        source: "",
                        status: "idle",
                        tile,
                    };
                    tenViewHistoryPreviewState.slots.set(slotId, record);
                }
                const image = tile.querySelector(
                    ".image-frame img, .image-container img"
                );
                if (!image) {
                    markTenViewHistorySlotWaiting(record, tile);
                    return;
                }
                const source = configureTenViewHistoryImage(
                    image,
                    index
                );
                if (!source) {
                    markTenViewHistorySlotWaiting(record, tile);
                    return;
                }

                if (
                    record.source === source
                    && record.image === image
                    && record.status === "loading"
                ) return;
                if (
                    record.source === source
                    && record.status === "ready"
                    && tenViewHistoryImageIsReady(image)
                ) {
                    stopTenViewHistorySlot(record);
                    record.image = image;
                    setTenViewHistoryTileState(record, tile, "ready");
                    return;
                }
                if (
                    record.source === source
                    && record.image === image
                    && record.status === "error"
                ) return;
                beginTenViewHistorySlotLoading(
                    record,
                    tile,
                    image,
                    source
                );
            });
            syncTenViewHistoryLoadingPhase();
        };
