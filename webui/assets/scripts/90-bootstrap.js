
        const observer = new MutationObserver(() => {
            installFooterItem();
            installUnifiedIcons();
            wireTopbar();
            wireModal();
            wirePresetCards();
            syncPresetSelection();
            installTabRouting();
            installGenerationRouting();
            syncGenerationConsoleFromUrl();
        });
        observer.observe(document.body, {childList: true, subtree: true});

        installFooterItem();
        installUnifiedIcons();
        wireTopbar();
        wireModal();
        wirePresetCards();
        syncPresetSelection();
        installTabRouting();
        installGenerationRouting();
        syncFromUrl();
        syncGenerationConsoleFromUrl();

        window.addEventListener("popstate", () => {
            const nextGenerationUid = currentAppUrl().searchParams.get("generation");
            if (nextGenerationUid !== activeGenerationRouteUid) {
                window.location.reload();
                return;
            }
            syncFromUrl();
            syncTabFromUrl();
            syncGenerationConsoleFromUrl();
        });
        document.addEventListener("keydown", (event) => {
            if (event.key === "Escape" && modal()?.classList.contains("rtx-open")) {
                closeModal();
            }
        });
    }
