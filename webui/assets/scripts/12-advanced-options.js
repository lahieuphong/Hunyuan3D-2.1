        let advancedOptionsExpanded = true;

        const syncAdvancedOptionsDisclosure = (expanded = advancedOptionsExpanded) => {
            const settingsTabs = document.getElementById("settings-tabs");
            const button = document.getElementById("advanced-settings-form-button");
            if (!settingsTabs || !button) return;

            advancedOptionsExpanded = expanded;
            settingsTabs.classList.toggle("is-advanced-collapsed", !expanded);
            button.setAttribute("aria-controls", "advanced-settings-form");
            button.setAttribute("aria-expanded", String(expanded));

            if (!button.querySelector(":scope > .advanced-options-chevron")) {
                button.insertAdjacentHTML(
                    "beforeend",
                    uiIconMarkup("chevronDown", "advanced-options-chevron")
                );
            }
        };

        const installAdvancedOptionsDisclosure = () => {
            const button = document.getElementById("advanced-settings-form-button");
            if (!button || !document.getElementById("advanced-settings-form")) return;

            syncAdvancedOptionsDisclosure();
            if (button.dataset.advancedOptionsWired === "true") return;

            button.dataset.advancedOptionsWired = "true";
            button.addEventListener("click", (event) => {
                const isSelected = button.getAttribute("aria-selected") === "true";
                if (!isSelected) {
                    syncAdvancedOptionsDisclosure(true);
                    return;
                }

                event.preventDefault();
                event.stopImmediatePropagation();
                syncAdvancedOptionsDisclosure(!advancedOptionsExpanded);
            }, true);
        };
