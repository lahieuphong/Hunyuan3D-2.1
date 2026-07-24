
        const hardwareModalViews = new Set(["hardware", "rtx3090"]);
        const hardwarePresetCardSelector = [
            ".rtx3090-profile-card[data-profile]",
            ".hardware-preset-grid [data-profile][data-mutates-generation-settings]",
        ].join(", ");
        let modalOpenedFromApp = false;
        let hardwareModalReturnFocus = null;

        const isHardwareModalView = (value) => hardwareModalViews.has(value);

        const setHardwareTriggerExpanded = (isOpen) => {
            [
                document.getElementById("app-rtx-profile"),
                document.getElementById(footerButtonId),
            ].forEach((trigger) => {
                trigger?.setAttribute("aria-expanded", String(isOpen));
            });
        };

        const setModalOpen = (isOpen, shouldFocusClose = false) => {
            const element = modal();
            if (!element) return;
            element.classList.toggle("rtx-open", isOpen);
            element.setAttribute("aria-hidden", String(!isOpen));
            document.body.classList.toggle("rtx3090-modal-open", isOpen);
            setHardwareTriggerExpanded(isOpen);
            if (isOpen && shouldFocusClose) {
                window.setTimeout(() => {
                    document.getElementById("rtx3090-modal-close")?.focus();
                }, 0);
            }
        };

        const renderPresetSelection = (activeProfile) => {
            const element = modal();
            if (!element) return;
            const knownProfile = Boolean(presetButtonIds[activeProfile]);

            element.querySelectorAll(hardwarePresetCardSelector).forEach((card) => {
                const isActive = knownProfile && card.dataset.profile === activeProfile;
                card.classList.toggle("is-selected", isActive);
                card.setAttribute("aria-pressed", String(isActive));
            });

            Object.entries(presetButtonIds).forEach(([profile, buttonId]) => {
                const button = document.getElementById(buttonId);
                if (!button) return;
                const isActive = knownProfile && profile === activeProfile;
                button.classList.toggle("rtx-preset-action-active", isActive);
                button.setAttribute("aria-pressed", String(isActive));
            });
        };

        const syncPresetSelection = () => {
            const activeProfile = modal()?.querySelector(
                ".rtx-preset-status[data-profile]"
            )?.dataset.profile;
            if (activeProfile) renderPresetSelection(activeProfile);
        };

        const wirePresetCards = () => {
            const element = modal();
            if (!element) return;

            element.querySelectorAll(hardwarePresetCardSelector).forEach((card) => {
                if (card.dataset.hardwarePresetWired === "true") return;
                card.dataset.hardwarePresetWired = "true";

                const applyCardPreset = () => {
                    if (isHistoryReviewRoute()) return;
                    const profile = card.dataset.profile;
                    const button = document.getElementById(presetButtonIds[profile]);
                    if (!button || button.disabled) return;
                    renderPresetSelection(profile);
                    button.click();
                };

                card.addEventListener("click", applyCardPreset);
                card.addEventListener("keydown", (event) => {
                    if (event.key !== "Enter" && event.key !== " ") return;
                    event.preventDefault();
                    applyCardPreset();
                });
            });

            Object.entries(presetButtonIds).forEach(([profile, buttonId]) => {
                const button = document.getElementById(buttonId);
                if (!button || button.dataset.hardwarePresetWired === "true") return;
                button.dataset.hardwarePresetWired = "true";
                button.addEventListener("click", (event) => {
                    if (isHistoryReviewRoute()) {
                        event.preventDefault();
                        syncPresetSelection();
                        return;
                    }
                    renderPresetSelection(profile);
                });
            });
        };

        const restoreHardwareModalFocus = () => {
            window.setTimeout(() => {
                const fallback = document.getElementById("app-rtx-profile")
                    || document.getElementById(footerButtonId);
                const target = hardwareModalReturnFocus?.isConnected
                    ? hardwareModalReturnFocus
                    : fallback;
                target?.focus();
                hardwareModalReturnFocus = null;
            }, 0);
        };

        const syncFromUrl = () => {
            const url = currentAppUrl();
            let view = url.searchParams.get("view");
            if (view === "rtx3090") {
                url.searchParams.set("view", "hardware");
                window.history.replaceState(window.history.state, "", url);
                view = "hardware";
            }
            const shouldOpen = view === "hardware";
            const wasOpen = modal()?.classList.contains("rtx-open") === true;
            setModalOpen(shouldOpen, shouldOpen && !wasOpen);
            if (!shouldOpen) {
                modalOpenedFromApp = false;
                if (wasOpen) restoreHardwareModalFocus();
            }
        };

        const openModal = (event) => {
            const url = currentAppUrl();
            const currentView = url.searchParams.get("view");
            hardwareModalReturnFocus = event?.currentTarget || document.activeElement;
            setGenerationHistoryOpen(false);
            if (isHardwareModalView(currentView)) {
                if (currentView !== "hardware") {
                    url.searchParams.set("view", "hardware");
                    window.history.replaceState(window.history.state, "", url);
                }
                setModalOpen(true, true);
                return;
            }
            url.searchParams.set("view", "hardware");
            window.history.pushState({}, "", url);
            modalOpenedFromApp = true;
            setModalOpen(true, true);
        };

        const closeModal = (restoreFocus = true) => {
            const url = currentAppUrl();
            const currentView = url.searchParams.get("view");
            setModalOpen(false);
            if (restoreFocus) restoreHardwareModalFocus();
            if (!isHardwareModalView(currentView)) {
                modalOpenedFromApp = false;
                return;
            }
            if (modalOpenedFromApp) {
                modalOpenedFromApp = false;
                window.history.back();
                return;
            }
            url.searchParams.delete("view");
            window.history.replaceState({}, "", url);
        };

        const trapHardwareModalFocus = (event) => {
            const element = modal();
            if (
                event.key !== "Tab"
                || !element?.classList.contains("rtx-open")
            ) return;
            const focusable = [...element.querySelectorAll(
                'button:not([disabled]), input:not([disabled]), select:not([disabled]), '
                + 'a[href], [role="button"][tabindex]:not([tabindex="-1"])'
            )].filter((node) => node.getClientRects().length > 0);
            if (!focusable.length) {
                event.preventDefault();
                element.querySelector(".rtx3090-modal-panel")?.focus();
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
        };
