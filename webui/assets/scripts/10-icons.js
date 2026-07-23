        const uiIconPaths = {
            box: '<path d="m3 8 9-5 9 5-9 5-9-5Z"></path><path d="m3 8v8l9 5 9-5V8"></path><path d="M12 13v8"></path>',
            terminal: '<path d="m5 7 5 5-5 5"></path><path d="M12 19h7"></path>',
            zap: '<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8Z"></path>',
            x: '<path d="M18 6 6 18"></path><path d="m6 6 12 12"></path>',
            check: '<path d="m5 12 4 4L19 6"></path>',
            info: '<circle cx="12" cy="12" r="9"></circle><path d="M12 11v5"></path><path d="M12 8h.01"></path>',
            warning: '<path d="m21.73 18-8-14a2 2 0 0 0-3.46 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"></path><path d="M12 9v4"></path><path d="M12 17h.01"></path>',
            memory: '<rect x="5" y="6" width="14" height="12" rx="2"></rect><path d="M9 10h6v4H9z"></path><path d="M8 3v3M12 3v3M16 3v3M8 18v3M12 18v3M16 18v3"></path>',
            wand: '<path d="m15 4 5 5L7 22H2v-5L15 4Z"></path><path d="m14 5 5 5"></path><path d="M6 3v4M4 5h4M19 15v4M17 17h4"></path>',
            code: '<path d="m8 9-3 3 3 3"></path><path d="m16 9 3 3-3 3"></path><path d="m14 5-4 14"></path>',
            settings: '<path d="M4 6h10M18 6h2M4 12h2M10 12h10M4 18h7M15 18h5"></path><circle cx="16" cy="6" r="2"></circle><circle cx="8" cy="12" r="2"></circle><circle cx="13" cy="18" r="2"></circle>',
            sun: '<circle cx="12" cy="12" r="4"></circle><path d="M12 2v2M12 20v2M4.93 4.93l1.42 1.42M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.42-1.42M17.66 6.34l1.41-1.41"></path>',
            moon: '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79Z"></path>',
            monitor: '<rect x="3" y="4" width="18" height="14" rx="2"></rect><path d="M8 22h8M12 18v4"></path>',
            history: '<path d="M3 12a9 9 0 1 0 3-6.7"></path><path d="M3 4v6h6"></path><path d="M12 7v5l3 2"></path>',
            rotate: '<path d="M3 12a9 9 0 1 0 3-6.7"></path><path d="M3 4v6h6"></path>',
            download: '<path d="M12 3v12"></path><path d="m7 10 5 5 5-5"></path><path d="M5 21h14"></path>',
            chevronDown: '<path d="m6 9 6 6 6-6"></path>',
        };

        const uiIconMarkup = (name, extraClass = "") => {
            const paths = uiIconPaths[name];
            if (!paths) return "";
            return '<svg class="ui-icon ' + extraClass + '" viewBox="0 0 24 24" aria-hidden="true">'
                + paths + '</svg>';
        };

        const syncGenerateButtonCopy = (forcedMultiView = null) => {
            const generateButton = document.getElementById("generate-3d-button");
            if (!generateButton) return;

            const isMultiView = typeof forcedMultiView === "boolean"
                ? forcedMultiView
                : document.querySelector(
                    '#prompt-mode-tabs button[data-tab-id="tab_mv_prompt"]'
                )?.getAttribute("aria-selected") === "true";
            let copy = generateButton.querySelector(".generate-button-copy");
            if (!copy) {
                copy = document.createElement("span");
                copy.className = "generate-button-copy";
                copy.append(document.createElement("strong"), document.createElement("small"));
                generateButton.replaceChildren();
                generateButton.insertAdjacentHTML("afterbegin", uiIconMarkup("wand", "ui-action-icon"));
                generateButton.append(copy);
            }
            const title = copy.querySelector("strong");
            const subtitle = copy.querySelector("small");
            const nextSubtitle = isMultiView
                ? "4 synchronized views"
                : "1 front image";
            if (title.textContent !== "Generate 3D") {
                title.textContent = "Generate 3D";
            }
            if (subtitle.textContent !== nextSubtitle) {
                subtitle.textContent = nextSubtitle;
            }
            generateButton.dataset.uiActionIcon = "true";
        };

        const installUnifiedIcons = () => {
            document.querySelectorAll(
                ".api-docs .banner-wrap:first-child h2 > img"
            ).forEach((image) => {
                if (image.dataset.uiIconWired === "true") return;
                const settingsIconSvg = (
                    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
                    + 'fill="none" stroke="#808080" stroke-linecap="round" '
                    + 'stroke-linejoin="round" stroke-width="1.5">'
                    + uiIconPaths.settings
                    + '</svg>'
                );
                image.dataset.uiIconWired = "true";
                image.src = "data:image/svg+xml;charset=utf-8,"
                    + encodeURIComponent(settingsIconSvg);
            });

            const settingsThemeIcons = [
                ["sun", "Light"],
                ["moon", "Dark"],
                ["monitor", "System"],
            ];
            document.querySelectorAll(
                ".api-docs .theme-buttons .theme-button > button"
            ).forEach((button, index) => {
                if (button.dataset.uiIconWired === "true") return;
                const [iconName, label] = settingsThemeIcons[index] ?? [];
                if (!iconName || !label) return;
                button.dataset.uiIconWired = "true";
                button.innerHTML = (
                    '<span class="settings-theme-content">'
                    + uiIconMarkup(iconName, "settings-theme-icon")
                    + '<span>' + label + '</span></span>'
                );
            });

            document.querySelectorAll(
                '.api-docs input[aria-label="Language"]'
            ).forEach((input) => {
                const iconWrap = input.closest(".secondary-wrap")
                    ?.querySelector(".icon-wrap");
                if (!iconWrap || iconWrap.dataset.uiIconWired === "true") return;
                iconWrap.dataset.uiIconWired = "true";
                iconWrap.innerHTML = uiIconMarkup(
                    "chevronDown",
                    "settings-language-icon"
                );
            });

            document.querySelectorAll("[data-ui-icon]").forEach((element) => {
                if (element.dataset.uiIconWired === "true") return;
                const iconName = element.dataset.uiIcon;
                if (!uiIconPaths[iconName]) return;
                element.dataset.uiIconWired = "true";
                element.innerHTML = uiIconMarkup(iconName);
            });

            syncGenerateButtonCopy();

            document.querySelectorAll('button.reset-button[data-testid="reset-button"]').forEach((button) => {
                if (button.dataset.uiIconWired === "true") return;
                button.dataset.uiIconWired = "true";
                button.innerHTML = uiIconMarkup("rotate");
            });

            document.querySelectorAll("#mesh-stats button.toggle").forEach((button) => {
                if (button.dataset.uiDisclosureIconWired === "true") return;
                button.dataset.uiDisclosureIconWired = "true";
                button.innerHTML = uiIconMarkup("chevronDown", "ui-disclosure-icon");
            });

            document.querySelectorAll(".file-preview td.download a").forEach((link) => {
                if (link.dataset.uiIconWired === "true") return;
                const filenameCell = link.closest("tr.file")?.querySelector("td.filename");
                const filename = filenameCell?.getAttribute("aria-label") ?? link.getAttribute("download") ?? "generated mesh";
                link.dataset.uiIconWired = "true";
                link.innerHTML = "Download " + uiIconMarkup("download");
                link.setAttribute("aria-label", "Download " + filename);
                link.setAttribute("title", "Download " + filename);
                if (filenameCell) filenameCell.setAttribute("title", filename);
            });

            const footerIcons = [
                ["button.show-api", "code"],
                ["button.settings", "settings"],
            ];
            footerIcons.forEach(([selector, iconName]) => {
                document.querySelectorAll("footer " + selector).forEach((element) => {
                    if (element.dataset.uiIconWired === "true") return;
                    element.dataset.uiIconWired = "true";
                    element.querySelector("img")?.remove();
                    element.insertAdjacentHTML("afterbegin", uiIconMarkup(iconName));
                });
            });
        };
