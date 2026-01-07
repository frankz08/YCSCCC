import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk, font, messagebox
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.patches as patches
from datetime import datetime
import random

NEIGHBORHOODS = {
    "Manhattan": [
        {"name": "Upper East Side", "grid_age": 45, "population": 220000, "substations": 4, "tree_cover": 0.25},
        {"name": "Harlem", "grid_age": 65, "population": 115000, "substations": 2, "tree_cover": 0.15},
        {"name": "Lower Manhattan", "grid_age": 55, "population": 95000, "substations": 5, "tree_cover": 0.10},
        {"name": "Midtown", "grid_age": 50, "population": 85000, "substations": 6, "tree_cover": 0.12},
    ],
    "Brooklyn": [
        {"name": "Bensonhurst", "grid_age": 68, "population": 140000, "substations": 2, "tree_cover": 0.22},
        {"name": "Sunset Park", "grid_age": 62, "population": 125000, "substations": 3, "tree_cover": 0.18},
        {"name": "Park Slope", "grid_age": 55, "population": 95000, "substations": 2, "tree_cover": 0.30},
    ],
    "Queens": [
        {"name": "Astoria", "grid_age": 58, "population": 160000, "substations": 3, "tree_cover": 0.22},
        {"name": "Flushing", "grid_age": 52, "population": 200000, "substations": 4, "tree_cover": 0.28},
        {"name": "Jamaica", "grid_age": 68, "population": 140000, "substations": 2, "tree_cover": 0.25},
    ],
    "Bronx": [
        {"name": "Riverdale", "grid_age": 48, "population": 55000, "substations": 2, "tree_cover": 0.35},
        {"name": "South Bronx", "grid_age": 72, "population": 125000, "substations": 2, "tree_cover": 0.12},
        {"name": "Fordham", "grid_age": 65, "population": 95000, "substations": 2, "tree_cover": 0.18},
    ],
}

class OutagePredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.train_model()
    
    def generate_training_data(self, n_samples=1000):
        data = []
        
        for _ in range(n_samples):
            grid_age = np.random.randint(30, 80)
            population = np.random.randint(50000, 250000)
            substations = np.random.randint(1, 7)
            tree_cover = np.random.uniform(0.05, 0.40)
            temperature = np.random.uniform(20, 105)
            wind_speed = np.random.uniform(0, 45)
            precipitation = np.random.uniform(0, 3)
            humidity = np.random.uniform(30, 95)
            hour = np.random.randint(0, 24)
            is_peak = 1 if 16 <= hour <= 20 else 0
            
            load_stress = (population / substations) / 50000
            if is_peak:
                load_stress *= 1.8
            
            risk_score = 0
            
            if grid_age > 60:
                risk_score += 0.3
            if load_stress > 1.5:
                risk_score += 0.25
            if temperature > 95 or wind_speed > 30 or precipitation > 1.5:
                risk_score += 0.3
            if tree_cover > 0.25 and wind_speed > 20:
                risk_score += 0.2
            if is_peak and temperature > 90:
                risk_score += 0.15
            
            outage = 1 if risk_score > 0.5 else 0
            
            data.append([
                grid_age, population, substations, tree_cover,
                temperature, wind_speed, precipitation, humidity,
                hour, is_peak, load_stress, outage
            ])
        
        df = pd.DataFrame(data, columns=[
            'grid_age', 'population', 'substations', 'tree_cover',
            'temperature', 'wind_speed', 'precipitation', 'humidity',
            'hour', 'is_peak', 'load_stress', 'outage'
        ])
        
        return df
    
    def train_model(self):
        df = self.generate_training_data()
        X = df.drop('outage', axis=1)
        y = df['outage']
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.accuracy = self.model.score(X_scaled, y)
    
    def predict_risk(self, grid_age, population, substations, tree_cover,
                    temperature, wind_speed, precipitation, humidity, hour):
        is_peak = 1 if 16 <= hour <= 20 else 0
        load_stress = (population / substations) / 50000
        if is_peak:
            load_stress *= 1.8
        
        features = np.array([[
            grid_age, population, substations, tree_cover,
            temperature, wind_speed, precipitation, humidity,
            hour, is_peak, load_stress
        ]])
        
        features_scaled = self.scaler.transform(features)
        risk_prob = self.model.predict_proba(features_scaled)[0][1]
        return risk_prob

class PowerOutageApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NYC Power Outage Risk Predictor")
        self.geometry("1200x850")
        self.configure(bg="#ffffff")
        
        self.colors = {
            "bg": "#ffffff",
            "panel": "#f8f9fa",
            "accent": "#e9ecef",
            "high_risk": "#dc3545",
            "med_risk": "#ffc107",
            "low_risk": "#28a745",
            "text": "#212529",
            "subtext": "#6c757d",
            "border": "#dee2e6"
        }
        
        self.predictor = OutagePredictor()
        
        self.temperature = tk.DoubleVar(value=85)
        self.wind_speed = tk.DoubleVar(value=15)
        self.precipitation = tk.DoubleVar(value=0.5)
        self.humidity = tk.DoubleVar(value=70)
        self.hour = tk.IntVar(value=18)
        
        self.all_neighborhoods = []
        for borough, hoods in NEIGHBORHOODS.items():
            for hood in hoods:
                hood['borough'] = borough
                self.all_neighborhoods.append(hood)
        
        self.create_widgets()
        self.update_predictions()
    
    def create_widgets(self):
        header = tk.Frame(self, bg=self.colors["bg"], height=100)
        header.pack(fill=tk.X, padx=40, pady=(30,0))
        header.pack_propagate(False)
        
        tk.Label(
            header,
            text="NYC Power Outage Risk Predictor",
            font=("Helvetica Neue", 28, "bold"),
            bg=self.colors["bg"],
            fg=self.colors["text"]
        ).pack(anchor="w", pady=(0,8))
        
        tk.Label(
            header,
            text="Machine learning predictions for grid infrastructure resilience",
            font=("Helvetica Neue", 14),
            bg=self.colors["bg"],
            fg=self.colors["subtext"]
        ).pack(anchor="w")
        
        tk.Frame(self, bg=self.colors["border"], height=1).pack(fill=tk.X, padx=40, pady=20)
        
        main = tk.Frame(self, bg=self.colors["bg"])
        main.pack(fill=tk.BOTH, expand=True, padx=40, pady=(0,40))
        
        left_panel = tk.Frame(main, bg=self.colors["panel"], width=340)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0,30))
        left_panel.pack_propagate(False)
        
        tk.Label(
            left_panel,
            text="Weather Conditions",
            font=("Helvetica Neue", 15, "bold"),
            bg=self.colors["panel"],
            fg=self.colors["text"]
        ).pack(anchor="w", padx=25, pady=(25,20))
        
        controls_frame = tk.Frame(left_panel, bg=self.colors["panel"], padx=25)
        controls_frame.pack(fill=tk.X, pady=(0,20))
        
        self.create_slider(controls_frame, "Temperature (°F)", self.temperature, 20, 110, 0)
        self.create_slider(controls_frame, "Wind Speed (mph)", self.wind_speed, 0, 50, 1)
        self.create_slider(controls_frame, "Precipitation (in)", self.precipitation, 0, 3, 2)
        self.create_slider(controls_frame, "Humidity (%)", self.humidity, 30, 100, 3)
        
        tk.Label(
            left_panel,
            text="Time of Day",
            font=("Helvetica Neue", 15, "bold"),
            bg=self.colors["panel"],
            fg=self.colors["text"]
        ).pack(anchor="w", padx=25, pady=(20,20))
        
        time_frame = tk.Frame(left_panel, bg=self.colors["panel"], padx=25)
        time_frame.pack(fill=tk.X, pady=(0,25))
        
        self.create_slider(time_frame, "Hour", self.hour, 0, 23, 0)
        
        tk.Button(
            left_panel,
            text="Update Analysis",
            command=self.update_predictions,
            font=("Helvetica Neue", 13, "bold"),
            bg="#007AFF",
            fg="#ffffff",
            activebackground="#0051D5",
            relief=tk.FLAT,
            padx=30,
            pady=14,
            cursor="hand2",
            borderwidth=0
        ).pack(pady=(0,25))
        
        tk.Frame(left_panel, bg=self.colors["border"], height=1).pack(fill=tk.X, padx=25, pady=20)
        
        info_frame = tk.Frame(left_panel, bg=self.colors["panel"], padx=25)
        info_frame.pack(fill=tk.X, pady=(0,25))
        
        tk.Label(
            info_frame,
            text="Model Information",
            font=("Helvetica Neue", 12, "bold"),
            bg=self.colors["panel"],
            fg=self.colors["text"]
        ).pack(anchor="w", pady=(0,10))
        
        tk.Label(
            info_frame,
            text=f"Accuracy: {self.predictor.accuracy*100:.1f}%",
            font=("Helvetica Neue", 11),
            bg=self.colors["panel"],
            fg=self.colors["subtext"]
        ).pack(anchor="w", pady=3)
        
        tk.Label(
            info_frame,
            text="Algorithm: Random Forest",
            font=("Helvetica Neue", 11),
            bg=self.colors["panel"],
            fg=self.colors["subtext"]
        ).pack(anchor="w", pady=3)
        
        tk.Label(
            info_frame,
            text="Features: Infrastructure, weather, time",
            font=("Helvetica Neue", 11),
            bg=self.colors["panel"],
            fg=self.colors["subtext"]
        ).pack(anchor="w", pady=3)
        
        right_panel = tk.Frame(main, bg=self.colors["bg"])
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        tk.Label(
            right_panel,
            text="Risk Assessment Map",
            font=("Helvetica Neue", 15, "bold"),
            bg=self.colors["bg"],
            fg=self.colors["text"]
        ).pack(anchor="w", pady=(0,15))
        
        self.fig1 = Figure(figsize=(8,5), facecolor=self.colors["bg"])
        self.ax1 = self.fig1.add_subplot(111)
        self.ax1.set_facecolor(self.colors["bg"])
        self.canvas1 = FigureCanvasTkAgg(self.fig1, master=right_panel)
        self.canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(0,25))
        
        tk.Label(
            right_panel,
            text="High-Risk Areas",
            font=("Helvetica Neue", 15, "bold"),
            bg=self.colors["bg"],
            fg=self.colors["text"]
        ).pack(anchor="w", pady=(0,15))
        
        risk_container = tk.Frame(right_panel, bg=self.colors["panel"], relief=tk.FLAT)
        risk_container.pack(fill=tk.BOTH, expand=True)
        
        self.risk_text = tk.Text(
            risk_container,
            font=("Menlo", 10),
            bg=self.colors["panel"],
            fg=self.colors["text"],
            height=12,
            relief=tk.FLAT,
            padx=20,
            pady=20,
            borderwidth=0,
            highlightthickness=0
        )
        self.risk_text.pack(fill=tk.BOTH, expand=True)
    
    def create_slider(self, parent, label, variable, from_, to, row):
        frame = tk.Frame(parent, bg=self.colors["panel"])
        frame.grid(row=row, column=0, sticky="ew", pady=8)
        parent.grid_columnconfigure(0, weight=1)
        
        tk.Label(
            frame,
            text=label,
            font=("Helvetica Neue", 12),
            bg=self.colors["panel"],
            fg=self.colors["text"],
            anchor="w"
        ).pack(anchor="w", pady=(0,6))
        
        slider_container = tk.Frame(frame, bg=self.colors["panel"])
        slider_container.pack(fill=tk.X)
        
        slider = tk.Scale(
            slider_container,
            from_=from_,
            to=to,
            variable=variable,
            orient=tk.HORIZONTAL,
            bg=self.colors["panel"],
            fg=self.colors["text"],
            highlightthickness=0,
            troughcolor=self.colors["accent"],
            resolution=0.1 if from_ < 10 else 1,
            showvalue=True,
            font=("Helvetica Neue", 11),
            relief=tk.FLAT,
            borderwidth=0
        )
        slider.pack(fill=tk.X)
    
    def update_predictions(self):
        temp = self.temperature.get()
        wind = self.wind_speed.get()
        precip = self.precipitation.get()
        humid = self.humidity.get()
        hour_val = self.hour.get()
        
        risks = []
        for hood in self.all_neighborhoods:
            risk = self.predictor.predict_risk(
                hood['grid_age'],
                hood['population'],
                hood['substations'],
                hood['tree_cover'],
                temp, wind, precip, humid, hour_val
            )
            risks.append({
                'name': hood['name'],
                'borough': hood['borough'],
                'risk': risk,
                'grid_age': hood['grid_age'],
                'substations': hood['substations']
            })
        
        risks.sort(key=lambda x: x['risk'], reverse=True)
        
        self.draw_risk_map(risks)
        self.update_risk_list(risks)
    
    def draw_risk_map(self, risks):
        self.ax1.clear()
        
        by_borough = {'Manhattan': [], 'Brooklyn': [], 'Queens': [], 'Bronx': []}
        for r in risks:
            by_borough[r['borough']].append(r)
        
        borough_positions = {
            'Manhattan': (0, 1),
            'Bronx': (0, 0),
            'Brooklyn': (1, 1),
            'Queens': (1, 0)
        }
        
        for borough, (col, row) in borough_positions.items():
            x_base = col * 5.5
            y_base = row * 4.5
            
            self.ax1.text(
                x_base + 2.75, y_base + 4,
                borough,
                ha='center', va='top',
                fontsize=13,
                color=self.colors['text'],
                weight='bold'
            )
            
            neighborhoods = by_borough[borough]
            for i, hood in enumerate(neighborhoods):
                y_pos = y_base + 3.2 - (i * 0.9)
                risk = hood['risk']
                
                if risk > 0.7:
                    color = self.colors['high_risk']
                elif risk > 0.4:
                    color = self.colors['med_risk']
                else:
                    color = self.colors['low_risk']
                
                rect = patches.Rectangle(
                    (x_base + 0.2, y_pos - 0.35),
                    5.1, 0.7,
                    linewidth=0,
                    facecolor=color,
                    alpha=0.85,
                    edgecolor='none'
                )
                self.ax1.add_patch(rect)
                
                self.ax1.text(
                    x_base + 0.5, y_pos,
                    hood['name'],
                    ha='left', va='center',
                    fontsize=10,
                    color='white',
                    weight='600'
                )
                
                self.ax1.text(
                    x_base + 5.0, y_pos,
                    f"{risk*100:.0f}%",
                    ha='right', va='center',
                    fontsize=11,
                    color='white',
                    weight='bold'
                )
        
        hour_24 = self.hour.get()
        if hour_24 == 0:
            time_str = "12:00 AM"
        elif hour_24 < 12:
            time_str = f"{hour_24}:00 AM"
        elif hour_24 == 12:
            time_str = "12:00 PM"
        else:
            time_str = f"{hour_24-12}:00 PM"
        
        self.ax1.set_xlim(-0.5, 11.5)
        self.ax1.set_ylim(-0.5, 9.5)
        self.ax1.axis('off')
        self.ax1.set_title(
            f'Risk levels at {time_str}',
            color=self.colors['text'],
            fontsize=14,
            pad=20,
            loc='left',
            weight='600'
        )
        
        self.fig1.tight_layout()
        self.canvas1.draw()
    
    def update_risk_list(self, risks):
        self.risk_text.delete("1.0", tk.END)
        
        self.risk_text.insert(tk.END, f"Current conditions: {self.temperature.get():.0f}°F")
        self.risk_text.insert(tk.END, f", {self.wind_speed.get():.0f} mph winds")
        self.risk_text.insert(tk.END, f", {self.precipitation.get():.1f}\" precipitation\n\n")
        
        high_risk = sum(1 for r in risks if r['risk'] > 0.7)
        med_risk = sum(1 for r in risks if 0.4 < r['risk'] <= 0.7)
        low_risk = sum(1 for r in risks if r['risk'] <= 0.4)
        
        self.risk_text.insert(tk.END, f"High risk: {high_risk}  |  ")
        self.risk_text.insert(tk.END, f"Medium: {med_risk}  |  ")
        self.risk_text.insert(tk.END, f"Low: {low_risk}\n\n")
        
        for i, risk_data in enumerate(risks[:5], 1):
            risk_pct = risk_data['risk'] * 100
            
            if risk_pct > 70:
                status = "Critical"
            elif risk_pct > 40:
                status = "Elevated"
            else:
                status = "Low"
            
            self.risk_text.insert(tk.END, f"{i}. {risk_data['name']}, {risk_data['borough']}\n")
            self.risk_text.insert(tk.END, f"   {risk_pct:.0f}% risk  •  {status}\n")
            self.risk_text.insert(tk.END, f"   Grid: {risk_data['grid_age']}yr old, {risk_data['substations']} substations\n")
            
            if risk_pct > 70:
                self.risk_text.insert(tk.END, f"   → Priority for infrastructure upgrade\n")
            
            self.risk_text.insert(tk.END, "\n")

if __name__ == "__main__":
    app = PowerOutageApp()
    app.mainloop()
