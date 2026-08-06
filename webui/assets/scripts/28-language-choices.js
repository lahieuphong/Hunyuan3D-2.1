
        const supportedAppLanguages = Object.freeze(
            [
                {code: "en", label: "English"},
                {code: "zh-CN", label: "简体中文"},
            ].filter(
                (language) => uiSupportedLocales.includes(language.code)
            )
        );
        const nativeLanguageInputSelector = (
            '.api-docs input[role="listbox"][aria-label="Language"], '
            + '.api-docs input[role="listbox"][aria-label="语言"]'
        );
        let languageBridgeSequence = 0;

        const finishLanguageBridge = (select, input, token) => {
            if (select?.isConnected) {
                select.disabled = false;
                select.removeAttribute("aria-busy");
            }
            if (input?.dataset.uiLanguageBridgeToken === token) {
                delete input.dataset.uiLanguageBridgeToken;
                delete input.dataset.uiLanguageNormalizing;
            }
        };

        const openNativeLanguageOptions = (input) => {
            try {
                input.focus({preventScroll: true});
            } catch {
                input.focus();
            }
            if (typeof PointerEvent === "function") {
                input.dispatchEvent(new PointerEvent("pointerdown", {
                    bubbles: true,
                    cancelable: true,
                    pointerType: "mouse",
                }));
            }
            input.dispatchEvent(new MouseEvent("mousedown", {
                bubbles: true,
                cancelable: true,
                view: window,
            }));
            input.click();
        };

        const chooseNativeLanguage = (input, select, language) => {
            if (!input || !language) return;
            const token = String(++languageBridgeSequence);
            input.dataset.uiLanguageBridgeToken = token;
            if (select) {
                select.disabled = true;
                select.setAttribute("aria-busy", "true");
            }

            let attempt = 0;
            const commit = () => {
                if (
                    !input.isConnected
                    || input.dataset.uiLanguageBridgeToken !== token
                ) {
                    finishLanguageBridge(select, input, token);
                    return;
                }

                const host = input.closest(".ui-language-restricted");
                const option = Array.from(host?.querySelectorAll(
                    'li[data-testid="dropdown-option"][role="option"]'
                ) ?? []).find(
                    (candidate) => candidate.getAttribute("aria-label")
                        === language.label
                );
                if (option) {
                    option.dispatchEvent(new MouseEvent("mousedown", {
                        bubbles: true,
                        cancelable: true,
                        view: window,
                    }));
                    finishLanguageBridge(select, input, token);
                    window.requestAnimationFrame(
                        installRestrictedLanguageChoices
                    );
                    return;
                }

                if (
                    attempt === 0
                    || input.getAttribute("aria-expanded") !== "true"
                ) {
                    openNativeLanguageOptions(input);
                }
                attempt += 1;
                if (attempt <= 12) {
                    window.setTimeout(commit, 16);
                    return;
                }

                const currentLanguage = supportedAppLanguages.find(
                    (candidate) => candidate.label === input.value
                );
                if (select && currentLanguage) {
                    select.value = currentLanguage.code;
                }
                finishLanguageBridge(select, input, token);
            };

            commit();
        };

        const createRestrictedLanguageControl = (host, wrap) => {
            const control = document.createElement("div");
            control.className = "ui-language-select-control";

            const select = document.createElement("select");
            select.className = "ui-language-select";
            select.dataset.testid = "restricted-language-select";
            select.setAttribute("aria-label", uiT("language.label"));
            select.autocomplete = "off";
            supportedAppLanguages.forEach((language) => {
                const option = document.createElement("option");
                option.value = language.code;
                option.textContent = language.label;
                option.lang = language.code;
                select.append(option);
            });
            select.addEventListener("change", () => {
                const language = supportedAppLanguages.find(
                    (candidate) => candidate.code === select.value
                );
                if (language) {
                    document.documentElement.lang = language.code;
                    try {
                        window.localStorage.setItem(
                            uiLocaleStorageKey,
                            language.code
                        );
                    } catch {
                        // Locale switching does not depend on browser storage.
                    }
                }
                const input = host.querySelector(
                    nativeLanguageInputSelector
                );
                chooseNativeLanguage(input, select, language);
            });

            control.append(select);
            control.insertAdjacentHTML(
                "beforeend",
                uiIconMarkup("chevronDown", "ui-language-select-icon")
            );
            wrap.append(control);
            return select;
        };

        const installRestrictedLanguageChoices = () => {
            document.querySelectorAll(nativeLanguageInputSelector).forEach(
                (input) => {
                    const wrap = input.closest(".wrap-inner");
                    const host = input.closest(".container");
                    const nativeControl = input.closest(".secondary-wrap");
                    if (!wrap || !host || !nativeControl) return;

                    host.classList.add("ui-language-restricted");
                    nativeControl.classList.add("ui-language-native-control");
                    input.readOnly = true;
                    input.tabIndex = -1;
                    input.setAttribute("aria-hidden", "true");

                    let select = wrap.querySelector(
                        ":scope > .ui-language-select-control "
                        + "> .ui-language-select"
                    );
                    if (!select) {
                        select = createRestrictedLanguageControl(host, wrap);
                    }

                    const nativeLanguage = supportedAppLanguages.find(
                        (candidate) => candidate.label === input.value
                    );
                    const desiredLocale = currentUiLocale();
                    const desiredLanguage = supportedAppLanguages.find(
                        (candidate) => candidate.code === desiredLocale
                    ) ?? supportedAppLanguages.find(
                        (candidate) => candidate.code === uiDefaultLocale
                    );
                    if (!desiredLanguage) return;
                    select.value = desiredLanguage.code;
                    select.setAttribute("aria-label", uiT("language.label"));

                    if (
                        nativeLanguage?.code !== desiredLanguage.code
                        && input.dataset.uiLanguageNormalizing !== "true"
                    ) {
                        input.dataset.uiLanguageNormalizing = "true";
                        chooseNativeLanguage(
                            input,
                            select,
                            desiredLanguage
                        );
                    }
                }
            );
        };
