# WebUI maintenance boundary

- `assets/styles/` owns presentation, grouped by screen/component responsibility.
- `assets/scripts/` owns browser interaction, grouped by runtime responsibility.
- `asset_manifest.py` is the single source of truth for CSS cascade and JavaScript composition order.
- `asset_loader.py` composes the fragments into the two strings consumed by Gradio.
- `templates.py` owns reusable HTML shell fragments.
- `gradio_app.py` still owns Gradio components, values, and callback wiring so model behavior stays unchanged.

JavaScript fragments intentionally share one lexical scope: the first opens the Gradio arrow function and the last closes it. Validate the composed bundle instead of running a fragment independently.

When changing the UI, preserve existing `elem_id`, HTML IDs/classes, and manifest order unless the related selectors and handlers are updated together.
