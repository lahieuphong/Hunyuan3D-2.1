"""Static HTML shell for the generation history dialog."""


def render_history_modal() -> str:
    """Render the modal shell populated by the browser-side history client."""
    return """
    <section id="generation-history-modal" class="generation-history-modal" aria-hidden="true">
        <div class="generation-history-panel" role="dialog" aria-modal="true" aria-labelledby="generation-history-title" tabindex="-1">
            <header class="generation-history-header">
                <div class="generation-history-heading">
                    <span class="generation-history-heading-icon ui-icon-slot" data-ui-icon="history" aria-hidden="true"></span>
                    <div>
                        <h2 id="generation-history-title" data-ui-i18n="history.title">Generation History</h2>
                        <p data-ui-i18n="history.subtitle">All 3D models saved locally on this machine</p>
                    </div>
                </div>
                <div class="generation-history-header-actions">
                    <span id="generation-history-count" class="generation-history-count" data-ui-i18n="history.model_count_other" data-ui-i18n-count="0">0 models</span>
                    <button id="generation-history-refresh" class="generation-history-icon-button" type="button" aria-label="Refresh history" title="Refresh history" data-ui-i18n-aria-label="history.refresh" data-ui-i18n-title="history.refresh">
                        <span class="ui-icon-slot" data-ui-icon="rotate" aria-hidden="true"></span>
                    </button>
                    <button id="generation-history-close" class="generation-history-icon-button" type="button" aria-label="Close generation history" data-ui-i18n-aria-label="history.close">
                        <svg class="generation-history-close-icon" width="100%" height="100%" viewBox="0 0 5 5" version="1.1" xmlns="http://www.w3.org/2000/svg" xml:space="preserve" style="fill: currentcolor; fill-rule: evenodd; clip-rule: evenodd; stroke-linejoin: round; stroke-miterlimit: 2;" aria-hidden="true" focusable="false">
                            <g>
                                <path d="M3.789,0.09C3.903,-0.024 4.088,-0.024 4.202,0.09L4.817,0.705C4.931,0.819 4.931,1.004 4.817,1.118L1.118,4.817C1.004,4.931 0.819,4.931 0.705,4.817L0.09,4.202C-0.024,4.088 -0.024,3.903 0.09,3.789L3.789,0.09Z"></path>
                                <path d="M4.825,3.797C4.934,3.907 4.934,4.084 4.825,4.193L4.193,4.825C4.084,4.934 3.907,4.934 3.797,4.825L0.082,1.11C-0.027,1.001 -0.027,0.823 0.082,0.714L0.714,0.082C0.823,-0.027 1.001,-0.027 1.11,0.082L4.825,3.797Z"></path>
                            </g>
                        </svg>
                    </button>
                </div>
            </header>
            <div class="generation-history-toolbar">
                <div>
                    <strong data-ui-i18n="history.generated_models">Generated models</strong>
                    <span id="generation-history-summary" aria-live="polite" data-ui-i18n="history.newest_first">Newest models appear first</span>
                </div>
                <span class="generation-history-local-badge">
                    <span class="generation-history-local-dot" aria-hidden="true"></span>
                    <span data-ui-i18n="history.saved_locally">Saved locally</span>
                </span>
            </div>
            <div class="generation-history-body">
                <div id="generation-history-loading" class="generation-history-grid generation-history-loading" role="status" aria-live="polite" aria-label="Loading generation history" data-ui-i18n-aria-label="history.loading_aria">
                    <div class="generation-history-skeleton"></div>
                    <div class="generation-history-skeleton"></div>
                    <div class="generation-history-skeleton"></div>
                    <div class="generation-history-skeleton"></div>
                </div>
                <div id="generation-history-error" class="generation-history-state" role="alert" hidden>
                    <span class="generation-history-state-icon ui-icon-slot" data-ui-icon="warning" aria-hidden="true"></span>
                    <strong data-ui-i18n="history.load_failed">History could not be loaded</strong>
                    <p data-ui-i18n="history.load_failed_help">Please check that the WebUI is still running, then try again.</p>
                    <button id="generation-history-retry" type="button" data-ui-i18n="history.try_again">Try again</button>
                </div>
                <div id="generation-history-empty" class="generation-history-state" role="status" aria-live="polite" hidden>
                    <span class="generation-history-state-icon ui-icon-slot" data-ui-icon="box" aria-hidden="true"></span>
                    <strong data-ui-i18n="history.empty_title">No generated models yet</strong>
                    <p data-ui-i18n="history.empty_help">Your completed 3D generations will appear here automatically.</p>
                </div>
                <div id="generation-history-list" class="generation-history-grid" role="list"></div>
            </div>
        </div>
    </section>
    """
