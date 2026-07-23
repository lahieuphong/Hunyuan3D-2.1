
        const observer = new MutationObserver(() => {
            installFooterItem();
            installUnifiedIcons();
            wireTopbar();
            wireModal();
            wirePresetCards();
            wireGenerationHistoryModal();
            syncPresetSelection();
            installTabRouting();
            installGenerationRouting();
            syncGenerationHistoryFromUrl();
            syncGenerationConsoleFromUrl();
        });
        observer.observe(document.body, {childList: true, subtree: true});

        installFooterItem();
        installUnifiedIcons();
        wireTopbar();
        wireModal();
        wirePresetCards();
        wireGenerationHistoryModal();
        syncPresetSelection();
        installTabRouting();
        installGenerationRouting();
        syncFromUrl();
        syncGenerationHistoryFromUrl();
        syncGenerationConsoleFromUrl();

        window.addEventListener("popstate", () => {
            const nextGenerationUid = currentAppUrl().searchParams.get("generation");
            if (nextGenerationUid !== activeGenerationRouteUid) {
                window.location.reload();
                return;
            }
            syncFromUrl();
            syncGenerationHistoryFromUrl();
            syncTabFromUrl();
            syncGenerationConsoleFromUrl();
        });
        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape" && generationHistoryModal()?.classList.contains("is-open")) {
                closeGenerationHistory(true);
                return;
            }
            if (event.key === "Escape" && modal()?.classList.contains("rtx-open")) {
                closeModal();
            }
        });
    }
