
        const tenViewPreviewRuntime = {
            controller: null,
            files: new Map(),
            objectUrls: new Map(),
            root: null,
        };
        const tenViewMaximumFileBytes = 30 * 1024 * 1024;

        const tenViewLooksLikeImage = (file) => (
            file instanceof File
            && (
                file.type.startsWith("image/")
                || /\.(?:jpe?g|png|webp)$/i.test(file.name)
            )
        );

        const releaseTenViewObjectUrl = (key) => {
            const objectUrl = tenViewPreviewRuntime.objectUrls.get(key);
            if (!objectUrl) return;
            URL.revokeObjectURL(objectUrl);
            tenViewPreviewRuntime.objectUrls.delete(key);
        };

        const updateTenViewCompleteness = (root) => {
            const cards = Array.from(root.querySelectorAll("[data-ten-view-card]"));
            const ready = cards.filter((card) => card.classList.contains("is-ready"));
            const count = ready.length;
            const total = cards.length;
            const countElement = root.querySelector("[data-ten-view-count]");
            const progress = root.querySelector("[data-ten-view-progress]");
            const fill = root.querySelector("[data-ten-view-progress-fill]");
            const placeholder = root.querySelector(".ten-view-generate-placeholder");
            const subtitle = placeholder?.querySelector("small");

            if (countElement) countElement.textContent = `${count} / ${total}`;
            if (progress) progress.setAttribute("aria-valuenow", String(count));
            if (fill) {
                fill.style.transform = `scaleX(${total ? count / total : 0})`;
            }
            root.classList.toggle("is-complete", total > 0 && count === total);
            if (subtitle) {
                subtitle.textContent = count === total
                    ? "10 views ready · pipeline connection is the next step"
                    : "10-view pipeline will be connected next";
            }
        };

        const showTenViewError = (card, message) => {
            const error = card.querySelector("[data-ten-view-error]");
            card.classList.remove("is-ready");
            card.classList.add("is-error");
            if (error) {
                error.textContent = message;
                error.hidden = false;
            }
            updateTenViewCompleteness(tenViewPreviewRuntime.root);
        };

        const resetTenViewCard = (card) => {
            const key = card.dataset.viewKey;
            const input = card.querySelector("[data-ten-view-input]");
            const preview = card.querySelector("[data-ten-view-preview]");
            const remove = card.querySelector("[data-ten-view-remove]");
            const error = card.querySelector("[data-ten-view-error]");

            releaseTenViewObjectUrl(key);
            tenViewPreviewRuntime.files.delete(key);
            card.classList.remove("is-ready", "is-error", "is-dragging");
            if (input) input.value = "";
            if (preview) {
                preview.removeAttribute("src");
                preview.hidden = true;
            }
            if (remove) remove.hidden = true;
            if (error) {
                error.textContent = "";
                error.hidden = true;
            }
            updateTenViewCompleteness(tenViewPreviewRuntime.root);
        };

        const previewTenViewFile = (card, file) => {
            const key = card.dataset.viewKey;
            const preview = card.querySelector("[data-ten-view-preview]");
            const remove = card.querySelector("[data-ten-view-remove]");
            const error = card.querySelector("[data-ten-view-error]");

            resetTenViewCard(card);
            if (!tenViewLooksLikeImage(file)) {
                showTenViewError(card, "Choose a PNG, JPG or WebP image");
                return;
            }
            if (file.size > tenViewMaximumFileBytes) {
                showTenViewError(card, "Image must be 30 MB or smaller");
                return;
            }
            if (!preview) {
                showTenViewError(card, "Preview is unavailable");
                return;
            }

            const objectUrl = URL.createObjectURL(file);
            tenViewPreviewRuntime.files.set(key, file);
            tenViewPreviewRuntime.objectUrls.set(key, objectUrl);
            preview.onload = () => {
                if (tenViewPreviewRuntime.objectUrls.get(key) !== objectUrl) return;
                card.classList.remove("is-error");
                card.classList.add("is-ready");
                preview.hidden = false;
                if (remove) remove.hidden = false;
                if (error) {
                    error.textContent = "";
                    error.hidden = true;
                }
                updateTenViewCompleteness(tenViewPreviewRuntime.root);
            };
            preview.onerror = () => {
                if (tenViewPreviewRuntime.objectUrls.get(key) !== objectUrl) return;
                releaseTenViewObjectUrl(key);
                tenViewPreviewRuntime.files.delete(key);
                preview.hidden = true;
                showTenViewError(card, "This image could not be previewed");
            };
            preview.src = objectUrl;
        };

        const disposeTenViewInputs = () => {
            tenViewPreviewRuntime.controller?.abort();
            tenViewPreviewRuntime.controller = null;
            tenViewPreviewRuntime.objectUrls.forEach((objectUrl) => {
                URL.revokeObjectURL(objectUrl);
            });
            tenViewPreviewRuntime.objectUrls.clear();
            tenViewPreviewRuntime.files.clear();
            tenViewPreviewRuntime.root = null;
        };

        const installTenViewInputs = () => {
            const root = document.querySelector("[data-ten-view-panel]");
            if (tenViewPreviewRuntime.root && tenViewPreviewRuntime.root !== root) {
                disposeTenViewInputs();
            }
            if (!root || tenViewPreviewRuntime.root === root) return;

            const controller = new AbortController();
            const listenerOptions = {signal: controller.signal};
            tenViewPreviewRuntime.controller = controller;
            tenViewPreviewRuntime.root = root;
            root.dataset.tenViewWired = "true";

            root.querySelectorAll("[data-ten-view-card]").forEach((card) => {
                const input = card.querySelector("[data-ten-view-input]");
                const dropzone = card.querySelector("[data-ten-view-dropzone]");
                const remove = card.querySelector("[data-ten-view-remove]");
                if (!input || !dropzone || !remove) return;

                input.addEventListener("change", () => {
                    const file = input.files?.[0];
                    if (file) previewTenViewFile(card, file);
                }, listenerOptions);
                dropzone.addEventListener("click", () => input.click(), listenerOptions);
                dropzone.addEventListener("keydown", (event) => {
                    if (event.key !== "Enter" && event.key !== " ") return;
                    event.preventDefault();
                    input.click();
                }, listenerOptions);
                dropzone.addEventListener("dragenter", (event) => {
                    event.preventDefault();
                    card.classList.add("is-dragging");
                }, listenerOptions);
                dropzone.addEventListener("dragover", (event) => {
                    event.preventDefault();
                    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
                }, listenerOptions);
                dropzone.addEventListener("dragleave", (event) => {
                    if (
                        event.relatedTarget instanceof Node
                        && dropzone.contains(event.relatedTarget)
                    ) return;
                    card.classList.remove("is-dragging");
                }, listenerOptions);
                dropzone.addEventListener("drop", (event) => {
                    event.preventDefault();
                    card.classList.remove("is-dragging");
                    const file = event.dataTransfer?.files?.[0];
                    if (file) previewTenViewFile(card, file);
                }, listenerOptions);
                remove.addEventListener("click", () => {
                    resetTenViewCard(card);
                    dropzone.focus();
                }, listenerOptions);
            });

            updateTenViewCompleteness(root);
        };
