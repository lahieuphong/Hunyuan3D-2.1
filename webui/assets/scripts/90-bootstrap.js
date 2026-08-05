
        const syncMountedUi = () => {
            installUiLocalization();
            installFooterItem();
            installUnifiedIcons();
            installAdvancedOptionsDisclosure();
            installSmoothThemeSwitching();
            installRestrictedLanguageChoices();
            installUiLocalization();
            installStableUploadPreviews();
            wireTopbar();
            wireModal();
            wirePresetCards();
            wireGenerationHistoryModal();
            syncPresetSelection();
            installTabRouting();
            installHistoryReviewMode();
            installTenViewHistoryLoading();
            installGenerationRouting();
            syncGenerationHistoryFromUrl();
            syncGenerationConsoleFromUrl();
        };
        let mountedUiSyncFrame = 0;
        const scheduleMountedUiSync = () => {
            if (mountedUiSyncFrame) return;
            mountedUiSyncFrame = window.requestAnimationFrame(() => {
                mountedUiSyncFrame = 0;
                syncMountedUi();
            });
        };
        const observer = new MutationObserver(scheduleMountedUiSync);
        observer.observe(document.body, {childList: true, subtree: true});

        syncMountedUi();
        syncFromUrl();

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
