
        const wireTopbar = () => {
            const apiButton = document.getElementById("app-api-docs");
            if (apiButton && apiButton.dataset.uiWired !== "true") {
                apiButton.dataset.uiWired = "true";
                apiButton.addEventListener("click", () => {
                    document.querySelector("footer button.show-api")?.click();
                });
            }

            const settingsButton = document.getElementById("app-theme-settings");
            if (settingsButton && settingsButton.dataset.uiWired !== "true") {
                settingsButton.dataset.uiWired = "true";
                settingsButton.addEventListener("click", () => {
                    document.querySelector("footer button.settings")?.click();
                });
            }

            const hardwareButton = document.getElementById("app-rtx-profile");
            if (hardwareButton && hardwareButton.dataset.uiWired !== "true") {
                hardwareButton.dataset.uiWired = "true";
                hardwareButton.setAttribute(
                    "aria-expanded",
                    String(modal()?.classList.contains("rtx-open") === true)
                );
                hardwareButton.addEventListener("click", openModal);
            }
        };

        const installFooterItem = () => {
            if (!modal()) return;
            const footer = Array.from(document.querySelectorAll("gradio-app footer, footer")).find(
                (element) => element.querySelector("button.show-api, a.built-with, button.settings")
            );
            if (!footer || document.getElementById(footerButtonId)) return;

            const builtWith = footer.querySelector("a.built-with");
            const settings = footer.querySelector("button.settings");
            const anchor = builtWith || settings;
            if (!anchor) return;

            const trigger = document.createElement("button");
            trigger.id = footerButtonId;
            trigger.type = "button";
            trigger.className = "rtx3090-footer-trigger";
            trigger.setAttribute("aria-haspopup", "dialog");
            trigger.setAttribute("aria-controls", modalId);
            trigger.setAttribute(
                "aria-expanded",
                String(modal()?.classList.contains("rtx-open") === true)
            );
            trigger.innerHTML = '<span class="rtx3090-footer-icon ui-icon-slot" data-ui-icon="memory"></span><span>GPU Presets · Cấu hình đề xuất</span>';
            trigger.addEventListener("click", openModal);

            const divider = document.createElement("div");
            divider.className = "divider rtx3090-footer-divider";
            divider.textContent = "·";

            footer.insertBefore(trigger, anchor);
            footer.insertBefore(divider, anchor);
        };

        const wireModal = () => {
            const element = modal();
            if (!element || element.dataset.rtxWired === "true") return;
            element.dataset.rtxWired = "true";
            element.setAttribute("role", "dialog");
            element.setAttribute("aria-modal", "true");
            element.setAttribute("aria-labelledby", "rtx3090-modal-title");
            element.setAttribute("aria-hidden", "true");
            const panel = element.querySelector(".rtx3090-modal-panel");
            panel?.setAttribute("tabindex", "-1");
            element.addEventListener("click", (event) => {
                if (event.target === element) closeModal();
            });
            element.addEventListener("keydown", trapHardwareModalFocus);
            document.getElementById("rtx3090-modal-close")?.addEventListener(
                "click",
                () => closeModal(true)
            );
        };
