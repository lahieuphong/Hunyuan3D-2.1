
        const setModalOpen = (isOpen, shouldFocusClose = false) => {
            const element = modal();
            if (!element) return;
            element.classList.toggle("rtx-open", isOpen);
            element.setAttribute("aria-hidden", String(!isOpen));
            document.body.classList.toggle("rtx3090-modal-open", isOpen);
            if (isOpen && shouldFocusClose) {
                document.getElementById("rtx3090-modal-close")?.focus();
            }
        };

        const renderPresetSelection = (activeProfile) => {
            const element = modal();
            if (!element) return;
            const knownProfile = Boolean(presetButtonIds[activeProfile]);

            element.querySelectorAll(".rtx3090-profile-card[data-profile]").forEach((card) => {
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

            element.querySelectorAll(".rtx3090-profile-card[data-profile]").forEach((card) => {
                if (card.dataset.rtxPresetWired === "true") return;
                card.dataset.rtxPresetWired = "true";

                const applyCardPreset = () => {
                    if (isHistoryReviewRoute()) return;
                    const profile = card.dataset.profile;
                    const button = document.getElementById(presetButtonIds[profile]);
                    if (!button) return;
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
                if (!button || button.dataset.rtxPresetWired === "true") return;
                button.dataset.rtxPresetWired = "true";
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

        let modalOpenedFromApp = false;

        const syncFromUrl = () => {
            const url = currentAppUrl();
            setModalOpen(url.searchParams.get("view") === "rtx3090");
        };

        const openModal = (event) => {
            const url = currentAppUrl();
            const shouldFocusClose = Boolean(event?.currentTarget?.matches?.(":focus-visible"));
            url.searchParams.set("view", "rtx3090");
            window.history.pushState({}, "", url);
            modalOpenedFromApp = true;
            setModalOpen(true, shouldFocusClose);
        };

        const closeModal = () => {
            const url = currentAppUrl();
            setModalOpen(false);
            if (url.searchParams.get("view") !== "rtx3090") {
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
