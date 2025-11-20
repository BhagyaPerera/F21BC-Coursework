# main.py or src/__main__.py

import tkinter as tk
from src.Gui.Gui import PSOANNGui


def main():
    # Create Tk window
    root = tk.Tk()

    # Launch our GUI class with the root window
    app = PSOANNGui(root)

    # Enter Tk main loop
    root.mainloop()


if __name__ == "__main__":
    main()
