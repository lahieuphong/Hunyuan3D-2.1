"""Small HTML templates used by the Gradio application shell."""


def render_topbar(
    brand_name: str,
    workspace_title: str,
    rtx_profile_action: str,
) -> str:
    """Render the application topbar without depending on runtime globals."""
    return f"""
    <header id="app-topbar" class="app-topbar">
        <div class="app-brand" aria-label="{brand_name}">
            <span class="app-brand-mark" aria-hidden="true">
                <img class="app-standard-logo" src="/favicon.ico" alt="" draggable="false">
            </span>
            <strong>{brand_name}</strong>
            <span class='app-version-badge'>v1.0</span>
        </div>
        <div class="app-title-block">
            <span class="app-title-mark" aria-hidden="true">
                <img class="app-standard-logo" src="/favicon.ico" alt="" draggable="false">
            </span>
            <div>
                <h1>{workspace_title}</h1>
                <p>Transform images into high-quality 3D assets with AI</p>
            </div>
        </div>
        <nav class="app-topbar-actions" aria-label="Application actions">
            <button id="app-api-docs" class="app-topbar-button" type="button">
                <span class="ui-icon-slot" data-ui-icon="code" aria-hidden="true"></span>
                <span>API Docs</span>
            </button>
            <button id="app-generation-history" class="app-topbar-button" type="button" aria-haspopup="dialog" aria-controls="generation-history-modal" aria-expanded="false">
                <span class="ui-icon-slot" data-ui-icon="history" aria-hidden="true"></span>
                <span>History</span>
            </button>
            <button id="app-theme-settings" class="app-topbar-button" type="button" aria-label="Settings">
                <span class="ui-icon-slot" data-ui-icon="settings" aria-hidden="true"></span>
                <span>Settings</span>
            </button>
            {rtx_profile_action}
        </nav>
    </header>
    """
