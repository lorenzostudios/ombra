"""
OMBRA - Video Perception Test Application
Main entry point launcher script.
"""

from src.app import App


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
