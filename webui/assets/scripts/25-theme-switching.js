
        const smoothThemeModes = ["light", "dark", "system"];
        const smoothThemeMediaQuery = window.matchMedia?.(
            "(prefers-color-scheme: dark)"
        ) ?? null;
        let smoothThemeClickWired = false;
        let smoothThemeMediaWired = false;
        let smoothThemeObservedBody = null;
        let smoothThemeBodyObserver = null;
        let smoothThemeTransition = null;
        let smoothThemeFallbackTimer = 0;

        const smoothThemeModeFromUrl = () => {
            const requested = currentAppUrl().searchParams.get("__theme");
            return requested === "light" || requested === "dark"
                ? requested
                : "system";
        };

        const resolveSmoothThemeMode = (mode) => {
            if (mode === "light" || mode === "dark") return mode;
            return smoothThemeMediaQuery?.matches ? "dark" : "light";
        };

        const smoothThemeButtons = () => Array.from(
            document.querySelectorAll(
                ".api-docs .theme-buttons .theme-button > button"
            )
        );

        const smoothThemeModeForButton = (button, index) => {
            const declaredMode = button?.dataset.uiThemeMode;
            return smoothThemeModes.includes(declaredMode)
                ? declaredMode
                : smoothThemeModes[index] ?? null;
        };

        const syncSmoothThemeControls = () => {
            const selectedMode = smoothThemeModeFromUrl();
            smoothThemeButtons().forEach((button, index) => {
                const mode = smoothThemeModeForButton(button, index);
                if (!mode) return;
                const item = button.closest(".theme-button");
                const isSelected = mode === selectedMode;
                button.dataset.uiThemeMode = mode;
                button.setAttribute("aria-pressed", String(isSelected));
                item?.classList.toggle("current-theme", isSelected);
                item?.classList.toggle("inactive-theme", !isSelected);
            });
        };

        const commitSmoothThemeMode = (mode, updateUrl) => {
            if (updateUrl) {
                const url = currentAppUrl();
                if (mode === "system") {
                    url.searchParams.delete("__theme");
                } else {
                    url.searchParams.set("__theme", mode);
                }
                window.history.replaceState(window.history.state, "", url);
            }

            const resolvedMode = resolveSmoothThemeMode(mode);
            document.body.classList.toggle("dark", resolvedMode === "dark");
            document.body.style.background = "var(--body-background-fill)";
            syncSmoothThemeControls();
            window.dispatchEvent(new CustomEvent("ui-theme-change", {
                detail: {mode, resolvedMode},
            }));
        };

        const runSmoothThemeTransition = (mode, updateUrl = true) => {
            const currentMode = smoothThemeModeFromUrl();
            const currentResolvedMode = document.body.classList.contains("dark")
                ? "dark"
                : "light";
            const nextResolvedMode = resolveSmoothThemeMode(mode);
            const commit = () => commitSmoothThemeMode(mode, updateUrl);

            if (
                updateUrl
                && mode === currentMode
                && currentResolvedMode === nextResolvedMode
            ) {
                syncSmoothThemeControls();
                return;
            }

            const reduceMotion = window.matchMedia?.(
                "(prefers-reduced-motion: reduce)"
            )?.matches === true;
            if (
                reduceMotion
                || currentResolvedMode === nextResolvedMode
            ) {
                commit();
                return;
            }

            smoothThemeTransition?.skipTransition?.();
            if (typeof document.startViewTransition === "function") {
                document.documentElement.classList.add("ui-theme-transition");
                try {
                    const transition = document.startViewTransition(commit);
                    smoothThemeTransition = transition;
                    const cleanup = () => {
                        if (smoothThemeTransition !== transition) return;
                        smoothThemeTransition = null;
                        document.documentElement.classList.remove(
                            "ui-theme-transition"
                        );
                    };
                    transition.finished.then(cleanup, cleanup);
                    return;
                } catch {
                    document.documentElement.classList.remove(
                        "ui-theme-transition"
                    );
                }
            }

            window.clearTimeout(smoothThemeFallbackTimer);
            document.documentElement.classList.add(
                "ui-theme-transition-fallback"
            );
            commit();
            smoothThemeFallbackTimer = window.setTimeout(() => {
                document.documentElement.classList.remove(
                    "ui-theme-transition-fallback"
                );
            }, 220);
        };

        const syncSmoothThemeFromUrl = (animate = false) => {
            const mode = smoothThemeModeFromUrl();
            const resolvedMode = resolveSmoothThemeMode(mode);
            const currentResolvedMode = document.body.classList.contains("dark")
                ? "dark"
                : "light";
            if (resolvedMode === currentResolvedMode) {
                syncSmoothThemeControls();
                return;
            }
            if (animate) {
                runSmoothThemeTransition(mode, false);
                return;
            }
            commitSmoothThemeMode(mode, false);
        };

        const handleSmoothThemeClick = (event) => {
            const target = event.target instanceof Element
                ? event.target
                : null;
            const item = target?.closest(
                ".api-docs .theme-buttons .theme-button"
            );
            if (!item) return;

            const buttons = smoothThemeButtons();
            const button = item.querySelector(":scope > button");
            const mode = smoothThemeModeForButton(
                button,
                buttons.indexOf(button)
            );
            if (!mode) return;

            event.preventDefault();
            event.stopPropagation();
            event.stopImmediatePropagation();
            runSmoothThemeTransition(mode);
        };

        const handleSmoothSystemThemeChange = () => {
            const mode = smoothThemeModeFromUrl();
            if (mode === "system") {
                runSmoothThemeTransition("system", false);
                return;
            }
            commitSmoothThemeMode(mode, false);
        };

        const installSmoothThemeBodyObserver = () => {
            if (!document.body || smoothThemeObservedBody === document.body) {
                return;
            }
            smoothThemeBodyObserver?.disconnect();
            smoothThemeObservedBody = document.body;
            smoothThemeBodyObserver = new MutationObserver(() => {
                const mode = smoothThemeModeFromUrl();
                if (mode === "system") {
                    syncSmoothThemeControls();
                    return;
                }
                const resolvedMode = resolveSmoothThemeMode(mode);
                const currentResolvedMode = document.body.classList.contains(
                    "dark"
                )
                    ? "dark"
                    : "light";
                if (currentResolvedMode !== resolvedMode) {
                    commitSmoothThemeMode(mode, false);
                    return;
                }
                syncSmoothThemeControls();
            });
            smoothThemeBodyObserver.observe(document.body, {
                attributes: true,
                attributeFilter: ["class"],
            });
        };

        const installSmoothThemeSwitching = () => {
            if (!smoothThemeClickWired) {
                document.addEventListener(
                    "click",
                    handleSmoothThemeClick,
                    true
                );
                smoothThemeClickWired = true;
            }
            if (smoothThemeMediaQuery && !smoothThemeMediaWired) {
                if (typeof smoothThemeMediaQuery.addEventListener === "function") {
                    smoothThemeMediaQuery.addEventListener(
                        "change",
                        handleSmoothSystemThemeChange
                    );
                } else {
                    smoothThemeMediaQuery.addListener(
                        handleSmoothSystemThemeChange
                    );
                }
                smoothThemeMediaWired = true;
            }
            installSmoothThemeBodyObserver();
            syncSmoothThemeFromUrl(false);
        };
