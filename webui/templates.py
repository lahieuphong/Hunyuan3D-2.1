"""Small HTML templates used by the Gradio application shell."""


_GENERATION_LOADING_COPY = {
    "preview": (
        "Generating 3D preview",
        "Building geometry and the interactive preview from your input views.",
        "box",
    ),
    "export": (
        "Preparing mesh for export",
        "The downloadable mesh will appear after generation is finalized.",
        "download",
    ),
    "statistics": (
        "Calculating mesh statistics",
        "Polygon, vertex and timing data will appear when the mesh is ready.",
        "terminal",
    ),
}


def render_generation_loading(kind: str) -> str:
    """Render a stable loading host that Gradio output updates cannot replace."""
    try:
        title, description, icon = _GENERATION_LOADING_COPY[kind]
    except KeyError as error:
        raise ValueError(f"Unknown generation loading kind: {kind}") from error

    return f"""
    <section class="generation-workspace-loading" data-generation-loading-kind="{kind}" data-state="idle" role="status" aria-live="polite" aria-atomic="true" aria-hidden="true">
        <div class="generation-loading-grid" aria-hidden="true"></div>
        <div class="generation-loading-card">
            <div class="generation-loading-visual" aria-hidden="true">
                <span class="generation-loading-orbit"></span>
                <span class="generation-loading-icon ui-icon-slot" data-ui-icon="{icon}"></span>
            </div>
            <div class="generation-loading-copy">
                <span class="generation-loading-phase">
                    <i aria-hidden="true"></i>
                    <span data-generation-loading-phase>Preparing request</span>
                </span>
                <h3 data-generation-loading-title>{title}</h3>
                <p data-generation-loading-message>{description}</p>
            </div>
            <div class="generation-loading-progress">
                <div class="generation-loading-progress-meta">
                    <span>Live generation progress</span>
                    <strong data-generation-loading-percent>0%</strong>
                </div>
                <div class="generation-loading-progress-track" role="progressbar" aria-label="Live generation progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
                    <span data-generation-loading-bar></span>
                </div>
            </div>
            <p class="generation-loading-console-hint">Live progress is available in Generation Console.</p>
        </div>
    </section>
    """


def render_topbar(
    brand_name: str,
    workspace_title: str,
    rtx_profile_action: str,
    *,
    show_api_docs_button: bool = True,
    show_history_button: bool = True,
    show_settings_button: bool = True,
    show_gpu_button: bool = True,
    dynamic_input_titles: bool = False,
    turbo_mode: bool = False,
) -> str:
    """Render the application topbar without depending on runtime globals."""
    api_docs_action = """
            <button id="app-api-docs" class="app-topbar-button" type="button">
                <span class="ui-icon-slot" data-ui-icon="code" aria-hidden="true"></span>
                <span>API Docs</span>
            </button>
    """ if show_api_docs_button else ""
    history_action = """
            <button id="app-generation-history" class="app-topbar-button" type="button" aria-haspopup="dialog" aria-controls="generation-history-modal" aria-expanded="false">
                <span class="ui-icon-slot" data-ui-icon="history" aria-hidden="true"></span>
                <span>History</span>
            </button>
    """ if show_history_button else ""
    settings_action = """
            <button id="app-theme-settings" class="app-topbar-button" type="button" aria-label="Settings">
                <span class="ui-icon-slot" data-ui-icon="settings" aria-hidden="true"></span>
                <span>Settings</span>
            </button>
    """ if show_settings_button else ""
    hardware_action = (
        rtx_profile_action
        if show_gpu_button
        else ""
    )
    actions = (
        api_docs_action
        + history_action
        + settings_action
        + hardware_action
    )
    action_navigation = f"""
        <nav class="app-topbar-actions" aria-label="Application actions">
            {actions}
        </nav>
    """ if actions.strip() else ""

    dynamic_input_titles_value = "true" if dynamic_input_titles else "false"
    turbo_mode_value = "true" if turbo_mode else "false"

    return f"""
    <header id="app-topbar" class="app-topbar" data-ui-dynamic-input-title="{dynamic_input_titles_value}" data-ui-turbo-mode="{turbo_mode_value}">
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
        {action_navigation}
    </header>
    """
