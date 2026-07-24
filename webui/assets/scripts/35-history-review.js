
        const historyReviewRouteUid = currentAppUrl().searchParams.get("generation");
        const historyReviewGuardSelector = [
            "#prompt-mode-tabs",
            ".generate-actions",
            "#generation-mode",
            "#decoding-mode",
            "#advanced-settings-form",
            "#rtx3090-modal .rtx3090-profile-grid",
            "#rtx3090-modal .rtx-preset-actions",
        ].join(", ");
        let historyReviewGuardInstalled = false;

        const historyReviewState = () => document.querySelector(
            '#history-review-state [data-history-review-active="true"]'
        );

        const isHistoryReviewRoute = () => Boolean(
            historyReviewRouteUid && historyReviewState()
        );

        const historyReviewGuardTarget = (event) => {
            if (!(event.target instanceof Element)) return null;
            return event.target.closest(historyReviewGuardSelector);
        };

        const installHistoryReviewMode = () => {
            const reviewState = historyReviewState();
            if (!historyReviewRouteUid || !reviewState) return;

            const expectedTabIndex = reviewState.dataset.inputMode === "four" ? 1 : 0;
            const expectedTabSlug = expectedTabIndex === 1
                ? "multi-view"
                : "single-view";
            const reviewUrl = currentAppUrl();
            if (reviewUrl.searchParams.get("tab") !== expectedTabSlug) {
                reviewUrl.searchParams.set("tab", expectedTabSlug);
                window.history.replaceState(window.history.state, "", reviewUrl);
            }
            const expectedTab = document.querySelectorAll(
                '#prompt-mode-tabs button[role="tab"]'
            )[expectedTabIndex];
            if (expectedTab?.getAttribute("aria-selected") !== "true") {
                expectedTab?.click();
            }

            document.body.classList.add("is-history-review");
            document.querySelectorAll(historyReviewGuardSelector).forEach((element) => {
                element.dataset.historyReadonly = "true";
                element.setAttribute("aria-disabled", "true");
                element.setAttribute(
                    "title",
                    "Saved History snapshot - generation inputs and settings are read-only."
                );
            });

            document.querySelectorAll([
                "#prompt-mode-tabs input",
                "#prompt-mode-tabs button:not([role='tab'])",
                ".generate-actions button",
                "#generation-mode input",
                "#generation-mode button",
                "#decoding-mode input",
                "#decoding-mode button",
                "#advanced-settings-form input",
                "#advanced-settings-form select",
                "#advanced-settings-form button",
                "#rtx3090-modal .rtx-preset-actions button",
            ].join(", ")).forEach((control) => {
                control.disabled = true;
                control.setAttribute("aria-disabled", "true");
            });

            document.querySelectorAll([
                "#prompt-mode-tabs button[role='tab']",
                "#rtx3090-modal .rtx3090-profile-card[data-profile]",
            ].join(", ")).forEach((control) => {
                control.setAttribute("aria-disabled", "true");
                control.tabIndex = -1;
            });

            const advancedButton = document.getElementById("advanced-settings-form-button");
            if (advancedButton && !advancedButton.querySelector(".history-readonly-badge")) {
                const badge = document.createElement("span");
                badge.className = "history-readonly-badge";
                badge.textContent = "History - read only";
                const chevron = advancedButton.querySelector(".advanced-options-chevron");
                advancedButton.insertBefore(badge, chevron || null);
            }

            if (historyReviewGuardInstalled) return;
            historyReviewGuardInstalled = true;

            [
                "pointerdown",
                "click",
                "change",
                "input",
                "dragenter",
                "dragover",
                "drop",
            ].forEach((eventName) => {
                document.addEventListener(eventName, (event) => {
                    if (!event.isTrusted || !historyReviewGuardTarget(event)) return;
                    event.preventDefault();
                    event.stopImmediatePropagation();
                }, true);
            });
            document.addEventListener("keydown", (event) => {
                if (
                    !event.isTrusted
                    || event.key === "Tab"
                    || event.key === "Escape"
                    || !historyReviewGuardTarget(event)
                ) return;
                event.preventDefault();
                event.stopImmediatePropagation();
            }, true);
        };
