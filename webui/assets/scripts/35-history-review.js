
        const historyReviewRouteUid = currentAppUrl().searchParams.get("generation");
        const historyReviewGuardSelector = [
            "#prompt-mode-tabs",
            ".generate-actions",
            "#generation-mode",
            "#decoding-mode",
            "#advanced-settings-form",
            "#hardware-profile-select",
            "#rtx3090-modal .rtx3090-profile-grid",
            "#rtx3090-modal .rtx-preset-actions",
            "[data-mutates-generation-settings]",
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

            const reviewMode = reviewState.dataset.inputMode;
            const expectedRoute = promptRouteForMode(reviewMode);
            const reviewUrl = currentAppUrl();
            if (reviewUrl.searchParams.get("tab") !== expectedRoute.slug) {
                reviewUrl.searchParams.set("tab", expectedRoute.slug);
                window.history.replaceState(window.history.state, "", reviewUrl);
            }

            // Gradio moves tabs that do not fit the narrow desktop rail into an
            // overflow menu. Select the saved route before read-only controls
            // are disabled so a multi-view History snapshot cannot be locked
            // on the default Single View panel.
            if (!promptRouteIsActive(expectedRoute)) {
                activatePromptTab(expectedRoute);
            }

            document.body.classList.add("is-history-review");
            document.querySelectorAll(historyReviewGuardSelector).forEach((element) => {
                element.dataset.historyReadonly = "true";
                element.dataset.uiI18nTitle = "history.readonly_title";
                element.setAttribute("aria-disabled", "true");
                const title = uiT("history.readonly_title");
                if (element.getAttribute("title") !== title) {
                    element.setAttribute("title", title);
                }
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
                "#hardware-profile-select input",
                "#hardware-profile-select select",
                "#hardware-profile-select button",
                "[data-mutates-generation-settings] input",
                "[data-mutates-generation-settings] select",
                "[data-mutates-generation-settings] button",
                "#rtx3090-modal .rtx-preset-actions button",
            ].join(", ")).forEach((control) => {
                control.disabled = true;
                control.setAttribute("aria-disabled", "true");
            });

            document.querySelectorAll([
                "#prompt-mode-tabs button[role='tab']",
                "#rtx3090-modal .rtx3090-profile-card[data-profile]",
                "#hardware-profile-select [role='combobox']",
                "[data-mutates-generation-settings][role='button']",
            ].join(", ")).forEach((control) => {
                control.setAttribute("aria-disabled", "true");
                control.tabIndex = -1;
            });

            const advancedButton = document.getElementById("advanced-settings-form-button");
            if (advancedButton) {
                let badge = advancedButton.querySelector(".history-readonly-badge");
                if (!badge) {
                    badge = document.createElement("span");
                    badge.className = "history-readonly-badge";
                    const chevron = advancedButton.querySelector(".advanced-options-chevron");
                    advancedButton.insertBefore(badge, chevron || null);
                }
                badge.dataset.uiI18n = "history.readonly_badge";
                const badgeCopy = uiT("history.readonly_badge");
                if (badge.textContent !== badgeCopy) {
                    badge.textContent = badgeCopy;
                }
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

        window.addEventListener("ui-language-change", () => {
            installHistoryReviewMode();
        });
