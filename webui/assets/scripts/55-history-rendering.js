
        let generationHistoryController = null;

        const generationHistoryElement = (id) => document.getElementById(id);
        const isMissingHistoryValue = (value) => (
            value === null || value === undefined || value === ""
        );

        const formatHistoryDate = (value) => {
            if (!value) return "Saved model";
            const parsed = new Date(value);
            if (Number.isNaN(parsed.getTime())) return "Saved model";
            return new Intl.DateTimeFormat(undefined, {
                dateStyle: "medium",
                timeStyle: "short",
            }).format(parsed);
        };

        const formatHistoryNumber = (value) => {
            if (isMissingHistoryValue(value)) return "—";
            const numeric = Number(value);
            if (!Number.isFinite(numeric)) return "—";
            return new Intl.NumberFormat(undefined, {
                notation: numeric >= 10000 ? "compact" : "standard",
                maximumFractionDigits: numeric >= 10000 ? 1 : 0,
            }).format(numeric);
        };

        const formatHistoryBytes = (value) => {
            if (isMissingHistoryValue(value)) return "—";
            const bytes = Number(value);
            if (!Number.isFinite(bytes) || bytes < 0) return "—";
            if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + " KB";
            return (bytes / (1024 * 1024)).toFixed(1) + " MB";
        };

        const formatHistorySeconds = (value) => {
            if (isMissingHistoryValue(value)) return "—";
            const seconds = Number(value);
            if (!Number.isFinite(seconds) || seconds < 0) return "—";
            return seconds.toFixed(seconds >= 10 ? 1 : 2) + " s";
        };

        const setGenerationHistoryState = (state) => {
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
                if (count) count.textContent = "Loading";
                if (summary) summary.textContent = "Reading saved 3D models...";
            } else if (state === "error") {
                const count = generationHistoryElement("generation-history-count");
                const summary = generationHistoryElement("generation-history-summary");
                if (count) count.textContent = "Unavailable";
                if (summary) summary.textContent = "Saved models could not be read";
            }
        };

        const createGenerationHistoryPlaceholder = () => {
            const placeholder = document.createElement("span");
            placeholder.className = "generation-history-placeholder";
            placeholder.insertAdjacentHTML("afterbegin", uiIconMarkup("box"));
            const label = document.createElement("span");
            label.textContent = "3D model";
            placeholder.append(label);
            return placeholder;
        };

        const createGenerationHistoryMetric = (value, label) => {
            const metric = document.createElement("div");
            metric.className = "generation-history-metric";
            const strong = document.createElement("strong");
            const caption = document.createElement("span");
            strong.textContent = value;
            caption.textContent = label;
            metric.append(strong, caption);
            return metric;
        };

        const openGenerationFromHistory = (item) => {
            if (item.legacy) {
                window.open(item.assets.viewer_url, "_blank", "noopener");
                return;
            }
            const url = currentAppUrl();
            if (item.input_mode === "four") {
                url.searchParams.set("tab", "multi-view");
            } else if (item.input_mode === "single") {
                url.searchParams.set("tab", "single-view");
            }
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
            preview.href = item.assets.viewer_url;
            preview.target = "_blank";
            preview.rel = "noopener";
            preview.setAttribute("aria-label", "Preview generation " + item.generation_uid);

            if (item.assets.thumbnail_url) {
                const image = document.createElement("img");
                image.src = item.assets.thumbnail_url;
                image.alt = "Input preview for generation " + item.generation_uid.slice(0, 8);
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
            const statusLabels = {
                completed: "Completed",
                failed: "Export saved",
                legacy: "Legacy mesh",
                processing: "Processing",
            };
            status.className = "generation-history-status is-" + statusKey;
            status.textContent = statusLabels[statusKey];
            preview.append(status);
            if (isCurrent) {
                const current = document.createElement("span");
                current.className = "generation-history-current";
                current.textContent = "Current";
                preview.append(current);
            }

            const body = document.createElement("div");
            body.className = "generation-history-card-body";
            const heading = document.createElement("div");
            heading.className = "generation-history-card-heading";
            const headingCopy = document.createElement("div");
            const title = document.createElement("h3");
            title.className = "generation-history-card-title";
            title.textContent = "Generation " + item.generation_uid.slice(0, 8).toUpperCase();
            title.title = item.generation_uid;
            const date = document.createElement("span");
            date.className = "generation-history-card-date";
            date.textContent = formatHistoryDate(item.completed_at || item.created_at);
            headingCopy.append(title, date);

            const model = document.createElement("span");
            model.className = "generation-history-model";
            model.textContent = item.model || (item.legacy ? "Saved mesh" : "Hunyuan3D");
            model.title = model.textContent;
            heading.append(headingCopy, model);

            const metrics = document.createElement("div");
            metrics.className = "generation-history-metrics";
            metrics.append(
                createGenerationHistoryMetric(
                    item.view_count ? String(item.view_count) : "—",
                    "Views"
                ),
                createGenerationHistoryMetric(
                    formatHistoryNumber(item.parameters.octree_resolution),
                    "Octree"
                ),
                createGenerationHistoryMetric(
                    formatHistorySeconds(item.statistics.seconds),
                    "Time"
                ),
                createGenerationHistoryMetric(
                    formatHistoryBytes(item.statistics.mesh_bytes),
                    "GLB"
                )
            );

            const actions = document.createElement("div");
            actions.className = "generation-history-actions";
            const open = document.createElement("button");
            open.className = "generation-history-action generation-history-open";
            open.type = "button";
            open.insertAdjacentHTML("afterbegin", uiIconMarkup("box"));
            const openLabel = document.createElement("span");
            openLabel.textContent = item.legacy
                ? "View 3D"
                : isCurrent ? "Open current" : "Open model";
            open.append(openLabel);
            open.addEventListener("click", () => openGenerationFromHistory(item));

            const download = document.createElement("a");
            download.className = "generation-history-action generation-history-download";
            download.href = item.assets.download_url;
            download.download = "white_mesh_" + item.generation_uid.slice(0, 8) + ".glb";
            download.insertAdjacentHTML("afterbegin", uiIconMarkup("download"));
            const downloadLabel = document.createElement("span");
            downloadLabel.textContent = "Download";
            download.append(downloadLabel);
            actions.append(open, download);

            body.append(heading, metrics, actions);
            article.append(preview, body);
            return article;
        };

        const renderGenerationHistory = (payload) => {
            const items = Array.isArray(payload?.items) ? payload.items : [];
            const total = Number.isFinite(Number(payload?.total)) ? Number(payload.total) : items.length;
            const count = generationHistoryElement("generation-history-count");
            const summary = generationHistoryElement("generation-history-summary");
            const list = generationHistoryElement("generation-history-list");
            if (count) count.textContent = total + (total === 1 ? " model" : " models");
            if (summary) {
                summary.textContent = total
                    ? "Showing " + items.length + " of " + total + " saved models"
                    : "Newest models appear first";
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
