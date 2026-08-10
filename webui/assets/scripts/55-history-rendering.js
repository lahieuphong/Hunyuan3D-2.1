
        let generationHistoryController = null;
        let generationHistoryLastPayload = null;
        let generationHistoryUiState = "idle";

        const generationHistoryElement = (id) => document.getElementById(id);
        const isMissingHistoryValue = (value) => (
            value === null || value === undefined || value === ""
        );

        const setGenerationHistoryCopy = (
            element,
            key,
            params = {}
        ) => {
            if (!element) return;
            element.dataset.uiI18n = key;
            const copy = uiT(key, params);
            if (element.textContent !== copy) element.textContent = copy;
        };

        const historyNumberFormat = (value, options = {}) => (
            new Intl.NumberFormat(currentUiLocale(), options).format(value)
        );

        const localizedGenerationViewerUrl = (rawUrl) => {
            if (!rawUrl) return rawUrl;
            try {
                const url = new URL(rawUrl, window.location.href);
                url.searchParams.set("lang", currentUiLocale());
                return url.href;
            } catch {
                return rawUrl;
            }
        };

        const formatHistoryDate = (value) => {
            if (!value) return uiT("history.saved_model");
            const parsed = new Date(value);
            if (Number.isNaN(parsed.getTime())) {
                return uiT("history.saved_model");
            }
            return new Intl.DateTimeFormat(currentUiLocale(), {
                dateStyle: "medium",
                timeStyle: "short",
            }).format(parsed);
        };

        const formatHistoryNumber = (value) => {
            if (isMissingHistoryValue(value)) return "—";
            const numeric = Number(value);
            if (!Number.isFinite(numeric)) return "—";
            return historyNumberFormat(numeric, {
                notation: numeric >= 10000 ? "compact" : "standard",
                maximumFractionDigits: numeric >= 10000 ? 1 : 0,
            });
        };

        const formatHistoryBytes = (value) => {
            if (isMissingHistoryValue(value)) return "—";
            const bytes = Number(value);
            if (!Number.isFinite(bytes) || bytes < 0) return "—";
            if (bytes < 1024 * 1024) {
                return uiT("history.size_kb", {
                    value: historyNumberFormat(bytes / 1024, {
                        maximumFractionDigits: 0,
                    }),
                });
            }
            return uiT("history.size_mb", {
                value: historyNumberFormat(bytes / (1024 * 1024), {
                    minimumFractionDigits: 1,
                    maximumFractionDigits: 1,
                }),
            });
        };

        const formatHistorySeconds = (value) => {
            if (isMissingHistoryValue(value)) return "—";
            const seconds = Number(value);
            if (!Number.isFinite(seconds) || seconds < 0) return "—";
            const fractionDigits = seconds >= 10 ? 1 : 2;
            return uiT("history.duration_seconds", {
                value: historyNumberFormat(seconds, {
                    minimumFractionDigits: fractionDigits,
                    maximumFractionDigits: fractionDigits,
                }),
            });
        };

        const setGenerationHistoryState = (state) => {
            generationHistoryUiState = state;
            const loading = generationHistoryElement("generation-history-loading");
            const error = generationHistoryElement("generation-history-error");
            const empty = generationHistoryElement("generation-history-empty");
            const list = generationHistoryElement("generation-history-list");
            if (loading) loading.hidden = state !== "loading";
            if (error) error.hidden = state !== "error";
            if (empty) empty.hidden = state !== "empty";
            if (list) list.hidden = state !== "content";
            if (state === "loading") {
                const count = generationHistoryElement("generation-history-count");
                const summary = generationHistoryElement("generation-history-summary");
                setGenerationHistoryCopy(count, "history.loading");
                setGenerationHistoryCopy(summary, "history.loading_summary");
            } else if (state === "error") {
                const count = generationHistoryElement("generation-history-count");
                const summary = generationHistoryElement("generation-history-summary");
                setGenerationHistoryCopy(count, "history.unavailable");
                setGenerationHistoryCopy(
                    summary,
                    "history.unavailable_summary"
                );
            }
        };

        const createGenerationHistoryPlaceholder = () => {
            const placeholder = document.createElement("span");
            placeholder.className = "generation-history-placeholder";
            placeholder.insertAdjacentHTML("afterbegin", uiIconMarkup("box"));
            const label = document.createElement("span");
            setGenerationHistoryCopy(label, "history.model_placeholder");
            placeholder.append(label);
            return placeholder;
        };

        const createGenerationHistoryMetric = (value, labelKey) => {
            const metric = document.createElement("div");
            metric.className = "generation-history-metric";
            const strong = document.createElement("strong");
            const caption = document.createElement("span");
            strong.textContent = value;
            setGenerationHistoryCopy(caption, labelKey);
            metric.append(strong, caption);
            return metric;
        };

        const openGenerationFromHistory = (item) => {
            if (item.legacy) {
                window.open(
                    localizedGenerationViewerUrl(item.assets.viewer_url),
                    "_blank",
                    "noopener"
                );
                return;
            }
            const url = currentAppUrl();
            const route = tabRoutes.find((candidate) => (
                candidate.mode === item.input_mode
            ));
            if (route) url.searchParams.set("tab", route.slug);
            url.searchParams.set("generation", item.generation_uid);
            url.searchParams.delete("view");
            window.location.assign(url);
        };

        const createGenerationHistoryCard = (item) => {
            const currentUid = currentAppUrl().searchParams.get("generation");
            const isCurrent = currentUid === item.generation_uid;
            const article = document.createElement("article");
            article.className = "generation-history-card" + (isCurrent ? " is-current" : "");
            article.dataset.generationUid = item.generation_uid;
            article.setAttribute("role", "listitem");

            const preview = document.createElement("a");
            preview.className = "generation-history-preview";
            preview.href = localizedGenerationViewerUrl(item.assets.viewer_url);
            preview.target = "_blank";
            preview.rel = "noopener";
            preview.dataset.uiI18nAriaLabel = "history.preview_generation";
            preview.setAttribute(
                "aria-label",
                uiT("history.preview_generation", {
                    uid: item.generation_uid,
                })
            );

            if (item.assets.thumbnail_url) {
                const image = document.createElement("img");
                image.src = item.assets.thumbnail_url;
                image.dataset.uiI18nAlt = "history.input_preview_alt";
                image.alt = uiT("history.input_preview_alt", {
                    uid: item.generation_uid.slice(0, 8),
                });
                image.loading = "lazy";
                image.addEventListener("error", () => {
                    image.replaceWith(createGenerationHistoryPlaceholder());
                }, {once: true});
                preview.append(image);
            } else {
                preview.append(createGenerationHistoryPlaceholder());
            }

            const status = document.createElement("span");
            const statusKey = item.legacy
                ? "legacy"
                : ["processing", "failed"].includes(item.status) ? item.status : "completed";
            const statusKeys = {
                completed: "history.status_completed",
                failed: "history.status_export_saved",
                legacy: "history.status_legacy_mesh",
                processing: "history.status_processing",
            };
            status.className = "generation-history-status is-" + statusKey;
            setGenerationHistoryCopy(status, statusKeys[statusKey]);
            preview.append(status);
            if (isCurrent) {
                const current = document.createElement("span");
                current.className = "generation-history-current";
                setGenerationHistoryCopy(current, "history.current");
                preview.append(current);
            }

            const body = document.createElement("div");
            body.className = "generation-history-card-body";
            const heading = document.createElement("div");
            heading.className = "generation-history-card-heading";
            const headingCopy = document.createElement("div");
            const title = document.createElement("h3");
            title.className = "generation-history-card-title";
            setGenerationHistoryCopy(title, "history.generation_title", {
                uid: item.generation_uid.slice(0, 8).toUpperCase(),
            });
            title.title = item.generation_uid;
            const date = document.createElement("span");
            date.className = "generation-history-card-date";
            date.textContent = formatHistoryDate(item.completed_at || item.created_at);
            headingCopy.append(title, date);

            const model = document.createElement("span");
            model.className = "generation-history-model";
            if (item.model) {
                model.textContent = item.model;
            } else if (item.legacy) {
                setGenerationHistoryCopy(model, "history.saved_mesh");
            } else {
                model.textContent = "Hunyuan3D";
            }
            model.title = model.textContent;
            heading.append(headingCopy, model);

            const metrics = document.createElement("div");
            metrics.className = "generation-history-metrics";
            metrics.append(
                createGenerationHistoryMetric(
                    item.view_count ? formatHistoryNumber(item.view_count) : "—",
                    "history.metric_views"
                ),
                createGenerationHistoryMetric(
                    formatHistoryNumber(item.parameters.octree_resolution),
                    "history.metric_octree"
                ),
                createGenerationHistoryMetric(
                    formatHistorySeconds(item.statistics.seconds),
                    "history.metric_time"
                ),
                createGenerationHistoryMetric(
                    formatHistoryBytes(item.statistics.mesh_bytes),
                    "history.metric_glb"
                )
            );

            const actions = document.createElement("div");
            actions.className = "generation-history-actions";
            const open = document.createElement("button");
            open.className = "generation-history-action generation-history-open";
            open.type = "button";
            open.insertAdjacentHTML("afterbegin", uiIconMarkup("box"));
            const openLabel = document.createElement("span");
            setGenerationHistoryCopy(
                openLabel,
                item.legacy
                    ? "history.view_3d"
                    : isCurrent
                        ? "history.open_current"
                        : "history.open_model"
            );
            open.append(openLabel);
            open.addEventListener("click", () => openGenerationFromHistory(item));

            const download = document.createElement("a");
            download.className = "generation-history-action generation-history-download";
            download.href = item.assets.download_url;
            const downloadPath = String(item.assets.download_url || "").split("?", 1)[0];
            const resolvedFilename = downloadPath.split("/").pop() || "model.glb";
            const downloadStem = resolvedFilename.replace(/\.glb$/i, "");
            download.download = downloadStem + "_" + item.generation_uid.slice(0, 8) + ".glb";
            download.insertAdjacentHTML("afterbegin", uiIconMarkup("download"));
            const downloadLabel = document.createElement("span");
            setGenerationHistoryCopy(downloadLabel, "history.download");
            download.append(downloadLabel);
            actions.append(open, download);

            body.append(heading, metrics, actions);
            article.append(preview, body);
            return article;
        };

        const renderGenerationHistory = (payload, cachePayload = true) => {
            if (cachePayload) generationHistoryLastPayload = payload;
            const items = Array.isArray(payload?.items) ? payload.items : [];
            const total = Number.isFinite(Number(payload?.total)) ? Number(payload.total) : items.length;
            const count = generationHistoryElement("generation-history-count");
            const summary = generationHistoryElement("generation-history-summary");
            const list = generationHistoryElement("generation-history-list");
            setGenerationHistoryCopy(
                count,
                total === 1
                    ? "history.model_count_one"
                    : "history.model_count_other",
                {count: historyNumberFormat(total)}
            );
            if (summary) {
                setGenerationHistoryCopy(
                    summary,
                    total
                        ? "history.showing_saved_models"
                        : "history.newest_first",
                    {
                        visible: historyNumberFormat(items.length),
                        total: historyNumberFormat(total),
                    }
                );
            }
            if (!list) return;
            list.replaceChildren();
            if (!items.length) {
                setGenerationHistoryState("empty");
                return;
            }
            const fragment = document.createDocumentFragment();
            items.forEach((item) => fragment.append(createGenerationHistoryCard(item)));
            list.append(fragment);
            setGenerationHistoryState("content");
        };

        const loadGenerationHistory = async () => {
            generationHistoryController?.abort();
            const controller = new AbortController();
            generationHistoryController = controller;
            const refresh = generationHistoryElement("generation-history-refresh");
            refresh?.classList.add("is-loading");
            if (refresh) refresh.disabled = true;
            setGenerationHistoryState("loading");
            try {
                const response = await fetch("/api/generation-history?limit=200", {
                    cache: "no-store",
                    headers: {Accept: "application/json"},
                    signal: controller.signal,
                });
                if (!response.ok) throw new Error("History request failed: " + response.status);
                renderGenerationHistory(await response.json());
            } catch (error) {
                if (error?.name !== "AbortError") {
                    setGenerationHistoryState("error");
                }
            } finally {
                if (generationHistoryController === controller) {
                    refresh?.classList.remove("is-loading");
                    if (refresh) refresh.disabled = false;
                }
            }
        };

        window.addEventListener("ui-language-change", () => {
            if (
                generationHistoryUiState === "loading"
                || generationHistoryUiState === "error"
            ) {
                setGenerationHistoryState(generationHistoryUiState);
                return;
            }
            if (generationHistoryLastPayload !== null) {
                renderGenerationHistory(generationHistoryLastPayload, false);
            }
        });
