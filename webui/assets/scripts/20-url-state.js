
        const tabRoutes = [
            {slug: "single-view", index: 0, tabId: "tab_single_prompt", label: "Single View"},
            {slug: "multi-view", index: 1, tabId: "tab_mv_prompt", label: "Multi View (1–4)"},
            {slug: "ten-view", index: 2, tabId: "tab_ten_prompt", label: "10 Views"},
        ];
        let tabRouteInitialized = false;

        const currentAppUrl = () => {
            const url = new URL(window.location.href);
            url.pathname = url.pathname.replace(/\/{2,}/g, "/");
            return url;
        };
        const canonicalInitialUrl = currentAppUrl();
        if (canonicalInitialUrl.href !== window.location.href) {
            window.history.replaceState(
                window.history.state,
                "",
                canonicalInitialUrl
            );
        }
        let activeGenerationRouteUid = currentAppUrl().searchParams.get("generation");

        const createGenerationUid = () => {
            if (window.crypto && typeof window.crypto.randomUUID === "function") {
                return window.crypto.randomUUID();
            }
            return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(
                /[xy]/g,
                (character) => {
                    const randomValue = Math.floor(Math.random() * 16);
                    const value = character === "x" ? randomValue : (randomValue & 0x3) | 0x8;
                    return value.toString(16);
                }
            );
        };
