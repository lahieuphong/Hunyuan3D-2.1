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
                        <h2 id="generation-history-title">Generation History</h2>
                        <p>All 3D models saved locally on this machine</p>
                    </div>
                </div>
                <div class="generation-history-header-actions">
                    <span id="generation-history-count" class="generation-history-count">0 models</span>
                    <button id="generation-history-refresh" class="generation-history-icon-button" type="button" aria-label="Refresh history" title="Refresh history">
                        <span class="ui-icon-slot" data-ui-icon="rotate" aria-hidden="true"></span>
                    </button>
                    <button id="generation-history-close" class="generation-history-icon-button" type="button" aria-label="Close generation history">
                        <span class="ui-icon-slot" data-ui-icon="x" aria-hidden="true"></span>
                    </button>
                </div>
            </header>
            <div class="generation-history-toolbar">
                <div>
                    <strong>Generated models</strong>
                    <span id="generation-history-summary" aria-live="polite">Newest models appear first</span>
                </div>
                <span class="generation-history-local-badge">
                    <span class="generation-history-local-dot" aria-hidden="true"></span>
                    Saved locally
                </span>
            </div>
            <div class="generation-history-body">
                <div id="generation-history-loading" class="generation-history-grid generation-history-loading" role="status" aria-live="polite" aria-label="Loading generation history">
                    <div class="generation-history-skeleton"></div>
                    <div class="generation-history-skeleton"></div>
                    <div class="generation-history-skeleton"></div>
                    <div class="generation-history-skeleton"></div>
                </div>
                <div id="generation-history-error" class="generation-history-state" role="alert" hidden>
                    <span class="generation-history-state-icon ui-icon-slot" data-ui-icon="warning" aria-hidden="true"></span>
                    <strong>History could not be loaded</strong>
                    <p>Please check that the WebUI is still running, then try again.</p>
                    <button id="generation-history-retry" type="button">Try again</button>
                </div>
                <div id="generation-history-empty" class="generation-history-state" role="status" aria-live="polite" hidden>
                    <span class="generation-history-state-icon ui-icon-slot" data-ui-icon="box" aria-hidden="true"></span>
                    <strong>No generated models yet</strong>
                    <p>Your completed 3D generations will appear here automatically.</p>
                </div>
                <div id="generation-history-list" class="generation-history-grid" role="list"></div>
            </div>
        </div>
    </section>
    """
