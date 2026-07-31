
        const normalizePromptTabLabel = (value) => String(value || "")
            .replace(/\s+/g, " ")
            .trim();

        const promptTabPanels = () => Array.from(
            document.querySelectorAll('#prompt-mode-tabs [role="tabpanel"]')
        ).slice(0, tabRoutes.length);

        const promptDirectTabButton = (route) => document.querySelector(
            '#prompt-mode-tabs button[role="tab"][data-tab-id="' + route.tabId + '"]'
        );

        const promptOverflowTabButton = (route) => Array.from(
            document.querySelectorAll("#prompt-mode-tabs .overflow-dropdown button")
        ).find((button) => (
            normalizePromptTabLabel(button.textContent) === route.label
        )) || null;

        const promptRouteIsActive = (route) => {
            const directButton = promptDirectTabButton(route);
            if (directButton?.getAttribute("aria-selected") === "true") {
                return true;
            }
            const panels = promptTabPanels();
            if (panels.length !== tabRoutes.length) return false;
            const panel = panels[route.index];
            return panel instanceof HTMLElement
                && window.getComputedStyle(panel).display !== "none";
        };

        const activatePromptTab = (route) => {
            if (promptRouteIsActive(route)) return true;
            const button = promptDirectTabButton(route)
                || promptOverflowTabButton(route);
            if (!button || button.disabled) return false;
            button.click();
            return true;
        };

        const promptDirectTabButtons = () => Array.from(
            document.querySelectorAll('#prompt-mode-tabs button[role="tab"]')
        );

        const promptOverflowTabButtons = () => Array.from(
            document.querySelectorAll("#prompt-mode-tabs .overflow-dropdown button")
        );

        const syncTabFromUrl = () => {
            if (!document.getElementById("prompt-mode-tabs")) return false;

            const url = currentAppUrl();
            const requestedSlug = url.searchParams.get("tab");
            const route = tabRoutes.find((item) => item.slug === requestedSlug) || tabRoutes[0];

            if (requestedSlug !== route.slug || url.href !== window.location.href) {
                url.searchParams.set("tab", route.slug);
                window.history.replaceState({}, "", url);
            }

            const activated = activatePromptTab(route);
            window.setTimeout(() => syncGenerateButtonCopy(route.slug), 0);
            return activated;
        };

        const installTabRouting = () => {
            const promptTabs = document.getElementById("prompt-mode-tabs");
            if (!promptTabs) return;

            promptDirectTabButtons().forEach((button) => {
                if (button.dataset.urlRouteWired === "true") return;
                const route = tabRoutes.find(
                    (candidate) => candidate.tabId === button.dataset.tabId
                );
                if (!route) return;
                button.dataset.urlRouteWired = "true";
                button.addEventListener("click", () => {
                    const slug = route.slug;
                    const url = currentAppUrl();
                    window.setTimeout(() => syncGenerateButtonCopy(slug), 0);
                    if (url.searchParams.get("tab") === slug) {
                        if (url.href !== window.location.href) {
                            window.history.replaceState({}, "", url);
                        }
                        return;
                    }
                    url.searchParams.set("tab", slug);
                    window.history.pushState({}, "", url);
                });
            });

            promptOverflowTabButtons().forEach((button) => {
                if (button.dataset.urlRouteWired === "true") return;
                const label = normalizePromptTabLabel(button.textContent);
                const route = tabRoutes.find(
                    (candidate) => candidate.label === label
                );
                if (!route) return;
                button.dataset.urlRouteWired = "true";
                button.addEventListener("click", () => {
                    const url = currentAppUrl();
                    window.setTimeout(() => syncGenerateButtonCopy(route.slug), 0);
                    if (url.searchParams.get("tab") === route.slug) {
                        if (url.href !== window.location.href) {
                            window.history.replaceState({}, "", url);
                        }
                        return;
                    }
                    url.searchParams.set("tab", route.slug);
                    window.history.pushState({}, "", url);
                });
            });

            const routeControlsReady = (
                promptTabPanels().length === tabRoutes.length
                && promptDirectTabButtons().length + promptOverflowTabButtons().length > 0
            );
            if (!routeControlsReady) return;

            if (!tabRouteInitialized) {
                tabRouteInitialized = true;
                [0, 100, 400].forEach((delay) => {
                    window.setTimeout(syncTabFromUrl, delay);
                });
            }
        };
