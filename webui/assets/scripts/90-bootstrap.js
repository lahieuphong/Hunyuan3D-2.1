
        const observer = new MutationObserver(() => {
            installFooterItem();
            installUnifiedIcons();
            installAdvancedOptionsDisclosure();
            installSmoothThemeSwitching();
            installStableUploadPreviews();
            wireTopbar();
            wireModal();
            wirePresetCards();
            wireGenerationHistoryModal();
            syncPresetSelection();
            installTabRouting();
            installHistoryReviewMode();
            installGenerationRouting();
            syncGenerationHistoryFromUrl();
            syncGenerationConsoleFromUrl();
        });
        observer.observe(document.body, {childList: true, subtree: true});

        installFooterItem();
        installUnifiedIcons();
        installAdvancedOptionsDisclosure();
        installSmoothThemeSwitching();
        installStableUploadPreviews();
        wireTopbar();
        wireModal();
        wirePresetCards();
        wireGenerationHistoryModal();
        syncPresetSelection();
        installTabRouting();
        installHistoryReviewMode();
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
            syncSmoothThemeFromUrl(false);
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
