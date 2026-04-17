import tkinter as tk
from tkinter import filedialog, messagebox
from main import run_pipeline

class ThreatDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Network Threat Detection System")
        self.root.geometry("500x400")

        self.file_path = None

        # Title
        tk.Label(root, text="Threat Detection System", font=("Arial", 16)).pack(pady=10)

        # Select File Button
        tk.Button(root, text="Select Dataset", command=self.select_file).pack(pady=5)

        # Train + Detect Button
        tk.Button(root, text="Train + Run Detection", command=self.train_and_detect).pack(pady=5)

        # Predict Button
        tk.Button(root, text="Run Detection", command=self.run_detection).pack(pady=5)

        # Output Box
        self.output = tk.Text(root, height=10)
        self.output.pack(pady=10)

    def log(self, msg):
        self.output.insert(tk.END, msg + "\n")
        self.output.see(tk.END)

    def select_file(self):
        self.file_path = filedialog.askopenfilename()
        self.log(f"Selected File: {self.file_path}")

    def train_and_detect(self):
        if not self.file_path:
            messagebox.showerror("Error", "Select a file first!")
            return

        try:
            run_pipeline(self.file_path, retrain=True, use_all_data=False)
            self.log("Model trained and detection completed.")
            self.log("Check report/security_report.txt and report/attack_results.png")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def run_detection(self):
        if not self.file_path:
            messagebox.showerror("Error", "Select a file first!")
            return

        try:
            run_pipeline(self.file_path, retrain=False, use_all_data=False)
            self.log("Detection completed using saved model.")
            self.log("Check report/security_report.txt and report/attack_results.png")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))


if __name__ == "__main__":
    root = tk.Tk()
    app = ThreatDetectionGUI(root)
    root.mainloop()

    # run it by : python gui.py