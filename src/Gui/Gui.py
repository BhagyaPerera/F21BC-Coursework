"""
Modern ANN + PSO GUI (Light Material Style)
Author: ChatGPT for Bhagya (2025)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import ttkbootstrap as tb
from ttkbootstrap.constants import *

import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.ann import ANNConfig
from src.pso import PSOConfig
from src.train.pipeline import run_pipeline


# ============================================================
# GUI CLASS
# ============================================================
class PSOANNGui:
    def __init__(self, root):
        self.root = root
        self.root.title("ANN + PSO Experimental Platform")
        self.root.geometry("1550x900")

        tb.Style("flatly")  # Light material theme

        self.hidden_layers = []
        self.experiments = []

        # Diff colors for runs
        self.color_map = plt.cm.tab10(np.linspace(0, 1, 5))

        self.build_layout()

    # ============================================================
    # Main Layout
    # ============================================================
    def build_layout(self):

        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)

        # ================================
        # LEFT SCROLLABLE SIDEBAR
        # ================================
        left_container = ttk.Frame(main)
        left_container.pack(side="left", fill="y")

        # Canvas for scrolling
        sidebar_canvas = tk.Canvas(left_container, width=1)
        sidebar_canvas.pack(side="left", fill="y", expand=True)

        # Scrollbar
        sidebar_scrollbar = ttk.Scrollbar(left_container, orient="vertical",
                                          command=sidebar_canvas.yview)
        sidebar_scrollbar.pack(side="right", fill="y")

        sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)
        sidebar_canvas.bind('<Configure>',
                            lambda e: sidebar_canvas.configure(
                                scrollregion=sidebar_canvas.bbox("all")
                            ))

        # Frame inside the canvas
        self.left = ttk.Frame(sidebar_canvas)
        sidebar_canvas.create_window((0, 0), window=self.left, anchor="nw")
        # Left side: configuration
        self.left = ttk.Frame(main, padding=10)
        self.left.pack(side="left", fill="y")

        # Right side: console + plots + table
        right = ttk.Frame(main, padding=10)
        right.pack(side="right", fill="both", expand=True)

        # ============================================================
        # ANN CONFIG PANEL
        # ============================================================
        ann_card = tb.Frame(self.left, padding=15, bootstyle="light")
        ann_card.pack(fill="x", pady=10)

        ttk.Label(ann_card, text="ANN Configuration",
                  font=("Segoe UI", 16, "bold")).pack()

        ttk.Label(ann_card, text="Input Dimension").pack(anchor="w")
        self.input_dim = tk.IntVar(value=8)
        ttk.Entry(ann_card, textvariable=self.input_dim).pack(fill="x")

        self.hl_container = ttk.LabelFrame(ann_card, text="Hidden Layers")
        self.hl_container.pack(fill="x", pady=10)

        ttk.Label(ann_card, text="Output Activation").pack(anchor="w")
        self.out_act = tk.StringVar(value="identity")

        ttk.Combobox(
            ann_card, textvariable=self.out_act,
            values=["identity", "relu", "tanh", "logistic"]
        ).pack(fill="x")


        tb.Button(
            ann_card, text="+ Add Hidden Layer", bootstyle="success",
            command=self.add_hidden_layer
        ).pack(fill="x", pady=5)

        # ============================================================
        # PSO CONFIG PANEL
        # ============================================================
        pso_card = tb.Frame(self.left, padding=15, bootstyle="light")
        pso_card.pack(fill="x", pady=10)

        ttk.Label(pso_card, text="PSO Configuration",
                  font=("Segoe UI", 16, "bold")).pack()

        self.swarm = tk.IntVar(value=25)
        self.iters = tk.IntVar(value=50)
        self.runs = tk.IntVar(value=5)

        self.alpha = tk.DoubleVar(value=0.1)
        self.beta = tk.DoubleVar(value=2.5)
        self.gamma = tk.DoubleVar(value=2.0)
        self.delta = tk.DoubleVar(value=0.0)

        self.bounds = tk.StringVar(value="-5,5")
        self.vclamp = tk.StringVar(value="-5,5")

        for label, var in [
            ("Swarm Size", self.swarm),
            ("Iterations", self.iters),
            ("Number of Runs", self.runs),
            ("Alpha", self.alpha),
            ("Beta", self.beta),
            ("Gamma", self.gamma),
            ("Delta", self.delta),
            ("Bounds (low,high)", self.bounds),
            ("Velocity Clamp", self.vclamp),
        ]:
            ttk.Label(pso_card, text=label).pack(anchor="w")
            ttk.Entry(pso_card, textvariable=var).pack(fill="x", pady=2)

        # ============================================================
        # ACTION BUTTONS
        # ============================================================
        action_card = tb.Frame(self.left, padding=15, bootstyle="light")
        action_card.pack(fill="x", pady=10)

        tb.Button(action_card, text="Run Experiment",
                  bootstyle="primary", command=self.run_experiment
                  ).pack(fill="x", pady=5)


        # ============================================================
        # CONSOLE
        # ============================================================
        console_frame = ttk.LabelFrame(right, text="Console Output")
        console_frame.pack(fill="x")

        self.console = tk.Text(console_frame, height=10, bg="#f8f9fa")
        self.console.pack(fill="x")

        self.progress = ttk.Progressbar(right, bootstyle="success-striped")
        self.progress.pack(fill="x", pady=5)

        # ============================================================
        # SUMMARY TABLE
        # ============================================================
        table_frame = ttk.LabelFrame(right, text="Run Summary")
        table_frame.pack(fill="x", pady=10)

        columns = ("run", "gbest", "train", "test", "config")
        self.table = ttk.Treeview(table_frame, columns=columns,
                                  show="headings", height=5)

        for col in columns:
            self.table.heading(col, text=col.capitalize())
            self.table.column(col, width=150, anchor="center")

        self.table.pack(fill="x")

        # ============================================================
        # PANEL PLOT AREA
        # ============================================================
        plot_frame = ttk.LabelFrame(right, text="Model Visualizations")
        plot_frame.pack(fill="both", expand=True)

        plot_frame.columnconfigure(0, weight=1)
        plot_frame.columnconfigure(1, weight=1)
        plot_frame.columnconfigure(2, weight=1)

        # Titles
        ttk.Label(plot_frame, text="PSO Convergence",
                  font=("Segoe UI", 12, "bold")).grid(row=0, column=0, pady=(5, 0))
        ttk.Label(plot_frame, text="Train Actual vs Predicted",
                  font=("Segoe UI", 12, "bold")).grid(row=0, column=1, pady=(5, 0))
        ttk.Label(plot_frame, text="Test Actual vs Predicted",
                  font=("Segoe UI", 12, "bold")).grid(row=0, column=2, pady=(5, 0))

        # PSO plot
        self.fig_conv, self.ax_conv = plt.subplots(figsize=(5, 4))
        self.canvas_conv = FigureCanvasTkAgg(self.fig_conv, master=plot_frame)
        self.canvas_conv.get_tk_widget().grid(row=1, column=0,
                                              sticky="nsew", padx=5, pady=5)

        # Train plot
        self.fig_train, self.ax_train = plt.subplots(figsize=(5, 4))
        self.canvas_train = FigureCanvasTkAgg(self.fig_train, master=plot_frame)
        self.canvas_train.get_tk_widget().grid(row=1, column=1,
                                               sticky="nsew", padx=5, pady=5)

        # Test plot
        self.fig_test, self.ax_test = plt.subplots(figsize=(5, 4))
        self.canvas_test = FigureCanvasTkAgg(self.fig_test, master=plot_frame)
        self.canvas_test.get_tk_widget().grid(row=1, column=2,
                                              sticky="nsew", padx=5, pady=5)

    # ============================================================
    # Hidden Layer Builder
    # ============================================================
    def add_hidden_layer(self):
        frame = ttk.Frame(self.hl_container)
        frame.pack(fill="x", pady=4)

        units = tk.IntVar(value=8)
        act = tk.StringVar(value="relu")

        ttk.Label(frame, text="Units").grid(row=0, column=0)
        ttk.Entry(frame, textvariable=units, width=6).grid(row=0, column=1)

        ttk.Label(frame, text="Act").grid(row=0, column=2)
        ttk.Combobox(frame, textvariable=act,
                     values=["relu", "tanh", "logistic"],
                     width=7).grid(row=0, column=3)

        tb.Button(frame, text="X", bootstyle="danger", width=2,
                  command=lambda: self.remove_hidden_layer(frame)
                  ).grid(row=0, column=4, padx=5)

        self.hidden_layers.append(
            {"frame": frame, "units": units, "activation": act}
        )

    def remove_hidden_layer(self, frame):
        for hl in self.hidden_layers:
            if hl["frame"] == frame:
                self.hidden_layers.remove(hl)
                break
        frame.destroy()

    # ============================================================
    # GUI Callback for Live Updates
    # ============================================================
    def gui_callback(self, iteration, best, run_index):

        color = self.color_map[run_index % len(self.color_map)]
        self.ax_conv.scatter(iteration, best, c=[color], s=12)
        self.canvas_conv.draw()

        self.console.insert(
            "end",
            f"[Run {run_index+1}] Iter {iteration}: gBest={best:.4f}\n"
        )
        self.console.see("end")
        self.progress["value"] = iteration
        self.root.update_idletasks()

    # ============================================================
    # Summary Table Update
    # ============================================================
    def update_summary_table(self, result, ann, pso):

        for child in self.table.get_children():
            self.table.delete(child)

        for i in range(len(result["train_maes"])):
            self.table.insert(
                "",
                "end",
                values=(
                    i + 1,
                    f"{result['gbest_vals'][i]:.4f}",
                    f"{result['train_maes'][i]:.4f}",
                    f"{result['test_maes'][i]:.4f}",
                    f"L={len(ann.hidden_layers)}, S={pso.swarm_size}, I={pso.iterations}"
                )
            )

    # ============================================================
    # Run Experiment
    # ============================================================
    def run_experiment(self):

        # Build ANN config
        ann = ANNConfig()
        ann.input_dim = self.input_dim.get()
        ann.output_dim = 1
        ann.output_activation = self.out_act.get()

        ann.hidden_layers = [
            {"units": hl["units"].get(), "activation": hl["activation"].get()}
            for hl in self.hidden_layers
        ]

        # PSO config
        low, high = map(float, self.bounds.get().split(","))
        v1, v2 = map(float, self.vclamp.get().split(","))

        pso = PSOConfig(
            swarm_size=self.swarm.get(),
            iterations=self.iters.get(),
            alpha=self.alpha.get(),
            beta=self.beta.get(),
            gamma=self.gamma.get(),
            delta=self.delta.get(),
            bounds=(low, high),
            v_clamp=(v1, v2),
        )

        # Reset plots
        self.ax_conv.clear()
        self.ax_conv.set_title("PSO Convergence")

        result = run_pipeline(
            ann_config=ann,
            pso_config=pso,
            runs=self.runs.get(),
            callback=self.gui_callback
        )
        self.experiments.append(result)

        self.update_summary_table(result, ann, pso)

        # --------------------------------------------------------
        # Plot TRAIN Actual vs Predicted
        # --------------------------------------------------------
        train = result["results"][0]["y_train"]
        train_pred = result["results"][0]["y_train_pred"]

        self.ax_train.clear()
        self.ax_train.scatter(train, train_pred, color="green", alpha=0.7, s=20)
        self.ax_train.plot(
            [min(train), max(train)],
            [min(train), max(train)],
            "r--"
        )
        self.ax_train.set_title("Train: Actual vs Predicted")
        self.ax_train.set_xlabel("Actual")
        self.ax_train.set_ylabel("Predicted")
        self.canvas_train.draw()

        # --------------------------------------------------------
        # Plot TEST Actual vs Predicted
        # --------------------------------------------------------
        test = result["results"][0]["y_test"]
        test_pred = result["results"][0]["y_test_pred"]

        self.ax_test.clear()
        self.ax_test.scatter(test, test_pred, color="purple", alpha=0.7, s=20)
        self.ax_test.plot(
            [min(test), max(test)],
            [min(test), max(test)],
            "r--"
        )
        self.ax_test.set_title("Test: Actual vs Predicted")
        self.ax_test.set_xlabel("Actual")
        self.ax_test.set_ylabel("Predicted")
        self.canvas_test.draw()

        self.console.insert("end", "--- Experiment Completed ---\n")

    # ============================================================
    # Compare Experiments (Mean Convergence Only)
    # ============================================================
    def compare_experiments(self):

        if len(self.experiments) < 2:
            return messagebox.showwarning("Need at least 2 Experiments",
                                          "Run at least 2 experiments before comparing.")

        self.ax_conv.clear()
        self.ax_conv.set_title("Comparison Across Experiments")

        for i, exp in enumerate(self.experiments):
            color = self.color_map[i % len(self.color_map)]

            if "histories" in exp:
                for h in exp["histories"]:
                    self.ax_conv.plot(h, color=color, alpha=0.4)

        self.canvas_conv.draw()
        self.console.insert("end", "--- Compared Experiments ---\n")

    # ============================================================
    # Export Plot
    # ============================================================
    def export_plot(self):
        path = filedialog.asksaveasfilename(defaultextension=".png")
        if path:
            self.fig_conv.savefig(path, dpi=180)
            messagebox.showinfo("Saved", "Plot exported as PNG.")

# ============================================================
# LAUNCHER
# ============================================================
def start_gui():
    root = tb.Window(themename="flatly")
    PSOANNGui(root)
    root.mainloop()
