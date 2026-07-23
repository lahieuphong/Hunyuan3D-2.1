
        const generationHistoryModal = () => document.getElementById("generation-history-modal");
        let generationHistoryOpenedFromApp = false;
        let generationHistoryReturnFocus = null;

        const setGenerationHistoryOpen = (isOpen, shouldFocusClose = false) => {
            const element = generationHistoryModal();
            if (!element) return;
            element.classList.toggle("is-open", isOpen);
            element.setAttribute("aria-hidden", String(!isOpen));
            document.body.classList.toggle("generation-history-modal-open", isOpen);
            document.getElementById("app-generation-history")?.setAttribute(
                "aria-expanded",
                String(isOpen)
            );
            if (isOpen && shouldFocusClose) {
                window.setTimeout(() => {
                    generationHistoryElement("generation-history-close")?.focus();
                }, 0);
            }
        };

        const syncGenerationHistoryFromUrl = () => {
            const shouldOpen = currentAppUrl().searchParams.get("view") === "history";
            const wasOpen = generationHistoryModal()?.classList.contains("is-open") === true;
            setGenerationHistoryOpen(shouldOpen, shouldOpen && !wasOpen);
            if (!shouldOpen) {
                generationHistoryOpenedFromApp = false;
                if (wasOpen) document.getElementById("app-generation-history")?.focus();
            }
            if (shouldOpen && !wasOpen) loadGenerationHistory();
        };

        const openGenerationHistory = (event) => {
            const url = currentAppUrl();
            const wasOpen = generationHistoryModal()?.classList.contains("is-open") === true;
            generationHistoryReturnFocus = event?.currentTarget || document.activeElement;
            setModalOpen(false);
            if (url.searchParams.get("view") === "history") {
                setGenerationHistoryOpen(true, true);
                if (!wasOpen) loadGenerationHistory();
                return;
            }
            url.searchParams.set("view", "history");
            window.history.pushState({}, "", url);
            generationHistoryOpenedFromApp = true;
            setGenerationHistoryOpen(true, true);
            loadGenerationHistory();
        };

        const closeGenerationHistory = (restoreFocus = true) => {
            const url = currentAppUrl();
            setGenerationHistoryOpen(false);
            if (restoreFocus) {
                window.setTimeout(() => {
                    const target = generationHistoryReturnFocus?.isConnected
                        ? generationHistoryReturnFocus
                        : document.getElementById("app-generation-history");
                    target?.focus();
                    generationHistoryReturnFocus = null;
                }, 0);
            }
            if (url.searchParams.get("view") !== "history") {
                generationHistoryOpenedFromApp = false;
                return;
            }
            if (generationHistoryOpenedFromApp) {
                generationHistoryOpenedFromApp = false;
                window.history.back();
                return;
            }
            url.searchParams.delete("view");
            window.history.replaceState({}, "", url);
        };

        const wireGenerationHistoryModal = () => {
            const trigger = document.getElementById("app-generation-history");
            if (trigger && trigger.dataset.historyWired !== "true") {
                trigger.dataset.historyWired = "true";
                trigger.addEventListener("click", openGenerationHistory);
            }

            const element = generationHistoryModal();
            if (!element || element.dataset.historyWired === "true") return;
            element.dataset.historyWired = "true";
            generationHistoryElement("generation-history-close")?.addEventListener(
                "click",
                () => closeGenerationHistory(true)
            );
            generationHistoryElement("generation-history-refresh")?.addEventListener(
                "click",
                loadGenerationHistory
            );
            generationHistoryElement("generation-history-retry")?.addEventListener(
                "click",
                loadGenerationHistory
            );
            element.addEventListener("click", (event) => {
                if (event.target === element) closeGenerationHistory();
            });
            element.addEventListener("keydown", (event) => {
                if (event.key !== "Tab" || !element.classList.contains("is-open")) return;
                const focusable = [...element.querySelectorAll(
                    'button:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])'
                )].filter((node) => node.getClientRects().length > 0);
                if (!focusable.length) {
                    event.preventDefault();
                    element.querySelector(".generation-history-panel")?.focus();
                    return;
                }
                const first = focusable[0];
                const last = focusable[focusable.length - 1];
                if (event.shiftKey && document.activeElement === first) {
                    event.preventDefault();
                    last.focus();
                } else if (!event.shiftKey && document.activeElement === last) {
                    event.preventDefault();
                    first.focus();
                }
            });
        };
