
        const promptTabButtons = () => Array.from(
            document.querySelectorAll('#prompt-mode-tabs button[role="tab"]')
        ).slice(0, tabRoutes.length);

        const syncTabFromUrl = () => {
            const buttons = promptTabButtons();
            if (buttons.length !== tabRoutes.length) return false;

            const url = currentAppUrl();
            const requestedSlug = url.searchParams.get("tab");
            const route = tabRoutes.find((item) => item.slug === requestedSlug) || tabRoutes[0];

            if (requestedSlug !== route.slug || url.href !== window.location.href) {
                url.searchParams.set("tab", route.slug);
                window.history.replaceState({}, "", url);
            }

            const target = buttons[route.index];
            if (target.getAttribute("aria-selected") !== "true") {
                target.click();
            }
            window.setTimeout(() => syncGenerateButtonCopy(route.index === 1), 0);
            return true;
        };

        const installTabRouting = () => {
            const buttons = promptTabButtons();
            if (buttons.length !== tabRoutes.length) return;

            buttons.forEach((button, index) => {
                if (button.dataset.urlRouteWired === "true") return;
                button.dataset.urlRouteWired = "true";
                button.addEventListener("click", () => {
                    const slug = tabRoutes[index].slug;
                    const url = currentAppUrl();
                    window.setTimeout(() => syncGenerateButtonCopy(index === 1), 0);
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

            if (!tabRouteInitialized) {
                tabRouteInitialized = true;
                [0, 100, 400].forEach((delay) => {
                    window.setTimeout(syncTabFromUrl, delay);
                });
            }
        };
