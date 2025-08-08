"""Entry point for launching the Gradio UI.

This small wrapper imports the Synergy Photo Editor module and runs its UI.
"""
import synergy_photo_editor_v_3 as app


def main() -> None:
    # Launch the UI. The underlying module decides whether to run the UI
    # based on environment variables and available dependencies.
    app.run_ui()


if __name__ == "__main__":
    main()
