
        const uiTranslationCatalog = Object.freeze(
            /*__UI_TRANSLATION_CATALOG__*/
        );
        const uiSupportedLocales = Object.freeze(["en", "zh-CN"]);
        const uiLocaleStorageKey = "hunyuan3d.ui-locale.v1";
        const uiTextBindings = new WeakMap();
        const uiAttributeBindings = new WeakMap();
        const uiFrameObservers = new WeakMap();
        const uiPendingTranslationRoots = new Set();
        let uiDocumentObserver = null;
        let uiForceFullTranslation = true;
        let uiLocaleObserver = null;
        let uiLastAppliedLocale = null;
        let uiTranslationFrame = 0;

        const normalizeUiLocale = (locale) => (
            String(locale || "").toLowerCase().startsWith("zh")
                ? "zh-CN"
                : "en"
        );

        const storedUiLocale = (() => {
            try {
                const stored = window.localStorage.getItem(uiLocaleStorageKey);
                return uiSupportedLocales.includes(stored) ? stored : null;
            } catch {
                return null;
            }
        })();
        if (storedUiLocale) {
            document.documentElement.lang = storedUiLocale;
        }

        const currentUiLocale = () => normalizeUiLocale(
            document.documentElement.lang || storedUiLocale || "en"
        );

        const formatUiTranslation = (template, params = {}) => (
            String(template).replace(/\{([A-Za-z0-9_]+)\}/g, (match, name) => (
                Object.prototype.hasOwnProperty.call(params, name)
                    ? String(params[name])
                    : match
            ))
        );

        const uiT = (key, params = {}, locale = currentUiLocale()) => {
            const entry = uiTranslationCatalog[key];
            if (!entry) return key;
            return formatUiTranslation(
                entry[normalizeUiLocale(locale)] ?? entry.en,
                params
            );
        };

        const normalizeUiText = (value) => String(value || "")
            .replace(/\s+/g, " ")
            .trim();
        const escapeUiPattern = (value) => value.replace(
            /[.*+?^${}()|[\]\\]/g,
            "\\$&"
        );
        const uiExactTranslationSources = new Map();
        const uiTemplateTranslationSources = [];

        Object.entries(uiTranslationCatalog).forEach(([key, entry]) => {
            Object.values(entry).forEach((translation) => {
                const normalized = normalizeUiText(translation);
                if (!normalized.includes("{")) {
                    if (!uiExactTranslationSources.has(normalized)) {
                        uiExactTranslationSources.set(normalized, key);
                    }
                    return;
                }
                const names = [];
                let cursor = 0;
                let pattern = "^";
                const placeholder = /\{([A-Za-z0-9_]+)\}/g;
                let match;
                while ((match = placeholder.exec(normalized)) !== null) {
                    pattern += escapeUiPattern(normalized.slice(cursor, match.index));
                    pattern += "(.+?)";
                    names.push(match[1]);
                    cursor = match.index + match[0].length;
                }
                pattern += escapeUiPattern(normalized.slice(cursor)) + "$";
                uiTemplateTranslationSources.push({
                    key,
                    names,
                    pattern: new RegExp(pattern, "u"),
                });
            });
        });

        const resolveUiTranslationSource = (value) => {
            const normalized = normalizeUiText(value);
            const exactKey = uiExactTranslationSources.get(normalized);
            if (exactKey) return {key: exactKey, params: {}};
            for (const source of uiTemplateTranslationSources) {
                const match = source.pattern.exec(normalized);
                if (!match) continue;
                const params = {};
                source.names.forEach((name, index) => {
                    params[name] = match[index + 1];
                });
                return {key: source.key, params};
            }
            return null;
        };

        const localizedUiBindingParams = (params) => (
            Object.fromEntries(Object.entries(params ?? {}).map(
                ([name, value]) => {
                    const nested = resolveUiTranslationSource(value);
                    return [
                        name,
                        nested && Object.keys(nested.params).length === 0
                            ? uiT(nested.key)
                            : value,
                    ];
                }
            ))
        );

        const splitUiWhitespace = (value) => {
            const leading = String(value).match(/^\s*/u)?.[0] ?? "";
            const trailing = String(value).match(/\s*$/u)?.[0] ?? "";
            const end = Math.max(leading.length, String(value).length - trailing.length);
            return {
                leading,
                trailing,
                core: String(value).slice(leading.length, end),
            };
        };

        const translateUiTextNode = (node) => {
            const parent = node.parentElement;
            if (!parent || parent.closest(
                "script, style, noscript, code, pre, [data-ui-i18n-ignore]"
            )) return;

            const parts = splitUiWhitespace(node.nodeValue ?? "");
            if (!parts.core.trim()) return;
            let binding = uiTextBindings.get(node);
            const normalizedCurrent = normalizeUiText(parts.core);
            if (!binding || normalizedCurrent !== normalizeUiText(binding.rendered)) {
                binding = resolveUiTranslationSource(parts.core);
                if (!binding) return;
            }

            const rendered = uiT(
                binding.key,
                localizedUiBindingParams(binding.params)
            );
            const nextValue = parts.leading + rendered + parts.trailing;
            binding.rendered = rendered;
            uiTextBindings.set(node, binding);
            if (node.nodeValue !== nextValue) node.nodeValue = nextValue;
        };

        const translateUiAttribute = (element, attribute) => {
            const value = element.getAttribute(attribute);
            if (!value) return;
            let bindings = uiAttributeBindings.get(element);
            if (!bindings) {
                bindings = new Map();
                uiAttributeBindings.set(element, bindings);
            }
            let binding = bindings.get(attribute);
            if (!binding || normalizeUiText(value) !== normalizeUiText(binding.rendered)) {
                binding = resolveUiTranslationSource(value);
                if (!binding) return;
            }
            const rendered = uiT(
                binding.key,
                localizedUiBindingParams(binding.params)
            );
            binding.rendered = rendered;
            bindings.set(attribute, binding);
            if (value !== rendered) element.setAttribute(attribute, rendered);
        };

        const translateUiDocument = (root) => {
            if (!root) return;
            const documentRoot = root.nodeType === Node.DOCUMENT_NODE
                ? root.documentElement
                : root;
            if (!documentRoot) return;
            const ownerDocument = documentRoot.ownerDocument ?? document;
            const walker = ownerDocument.createTreeWalker(
                documentRoot,
                NodeFilter.SHOW_TEXT
            );
            let textNode = walker.nextNode();
            while (textNode) {
                translateUiTextNode(textNode);
                textNode = walker.nextNode();
            }

            const elements = documentRoot.matches?.("*")
                ? [documentRoot, ...documentRoot.querySelectorAll("*")]
                : [...documentRoot.querySelectorAll("*")];
            elements.forEach((element) => {
                [
                    "alt",
                    "aria-description",
                    "aria-label",
                    "aria-valuetext",
                    "placeholder",
                    "title",
                ].forEach((attribute) => {
                    if (element.hasAttribute(attribute)) {
                        translateUiAttribute(element, attribute);
                    }
                });
            });
        };

        const queueUiTranslationRoot = (node) => {
            const root = node?.nodeType === Node.TEXT_NODE
                ? node.parentElement
                : node;
            if (!root) return;
            for (const pending of uiPendingTranslationRoots) {
                if (pending === root || pending.contains?.(root)) return;
                if (root.contains?.(pending)) {
                    uiPendingTranslationRoots.delete(pending);
                }
            }
            uiPendingTranslationRoots.add(root);
        };

        const queueUiMutationRecords = (records) => {
            records.forEach((record) => {
                if (record.type === "childList") {
                    record.addedNodes.forEach(queueUiTranslationRoot);
                    return;
                }
                queueUiTranslationRoot(record.target);
            });
            if (uiPendingTranslationRoots.size) scheduleUiTranslation();
        };

        const observeUiFrame = (frame) => {
            if (frame.dataset.uiI18nFrameWired !== "true") {
                frame.dataset.uiI18nFrameWired = "true";
                frame.addEventListener("load", () => observeUiFrame(frame));
            }
            let frameDocument;
            try {
                frameDocument = frame.contentDocument;
            } catch {
                return;
            }
            if (!frameDocument?.body) return;
            translateUiDocument(frameDocument);
            const previous = uiFrameObservers.get(frame);
            if (previous?.document === frameDocument) return;
            previous?.observer.disconnect();
            const observer = new MutationObserver(queueUiMutationRecords);
            observer.observe(frameDocument.body, {
                attributes: true,
                attributeFilter: [
                    "alt",
                    "aria-description",
                    "aria-label",
                    "aria-valuetext",
                    "placeholder",
                    "title",
                ],
                characterData: true,
                childList: true,
                subtree: true,
            });
            uiFrameObservers.set(frame, {document: frameDocument, observer});
        };

        const translateMountedUi = () => {
            translateUiDocument(document.body);
            document.querySelectorAll("iframe").forEach(observeUiFrame);
        };

        const translatePendingUi = () => {
            const roots = [...uiPendingTranslationRoots];
            uiPendingTranslationRoots.clear();
            roots.forEach((root) => {
                if (
                    root.nodeType !== Node.DOCUMENT_NODE
                    && root.isConnected === false
                ) return;
                translateUiDocument(root);
                const element = root.nodeType === Node.DOCUMENT_NODE
                    ? root.documentElement
                    : root;
                if (element?.matches?.("iframe")) observeUiFrame(element);
                element?.querySelectorAll?.("iframe").forEach(observeUiFrame);
            });
        };

        function scheduleUiTranslation() {
            if (uiTranslationFrame) return;
            uiTranslationFrame = window.requestAnimationFrame(() => {
                uiTranslationFrame = 0;
                installUiLocalization();
            });
        }

        const installUiLocalization = () => {
            if (!uiLocaleObserver) {
                uiLocaleObserver = new MutationObserver(scheduleUiTranslation);
                uiLocaleObserver.observe(document.documentElement, {
                    attributes: true,
                    attributeFilter: ["lang"],
                });
            }
            if (!uiDocumentObserver && document.body) {
                uiDocumentObserver = new MutationObserver(
                    queueUiMutationRecords
                );
                uiDocumentObserver.observe(document.body, {
                    attributes: true,
                    attributeFilter: [
                        "alt",
                        "aria-description",
                        "aria-label",
                        "aria-valuetext",
                        "placeholder",
                        "title",
                    ],
                    characterData: true,
                    childList: true,
                    subtree: true,
                });
            }
            const locale = currentUiLocale();
            if (document.body) document.body.dataset.uiLocale = locale;
            if (locale !== uiLastAppliedLocale) {
                uiLastAppliedLocale = locale;
                uiForceFullTranslation = true;
                try {
                    window.localStorage.setItem(uiLocaleStorageKey, locale);
                } catch {
                    // The UI still works when browser storage is unavailable.
                }
                window.dispatchEvent(new CustomEvent("ui-language-change", {
                    detail: {locale},
                }));
            }
            if (uiForceFullTranslation) {
                uiForceFullTranslation = false;
                uiPendingTranslationRoots.clear();
                translateMountedUi();
            } else {
                translatePendingUi();
            }
        };
