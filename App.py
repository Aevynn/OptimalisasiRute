import customtkinter as ctk
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import json
import numpy as np
import folium
import osmnx as ox
import networkx as nx
import webbrowser
import os

def load_locations_from_json(file_name="locations.json"):
    try:
        with open(file_name, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return {"CV. Sentosa Sumber Mukjizat": [-0.024033911619479945, 109.33987665422293]}
    except json.JSONDecodeError as e:
        messagebox.showerror("Kesalahan", f"Kesalahan membaca JSON: {e}")
        return {}

# Save locations to JSON
def save_locations_to_json(file_name="locations.json"):
    try:
        with open(file_name, "w") as file:
            json.dump(locations, file, indent=4)
        messagebox.showinfo("Info", "Lokasi berhasil disimpan!")
    except Exception as e:
        messagebox.showerror("Kesalahan", f"Kesalahan saat menyimpan lokasi: {e}")

# Add a new location
def add_location():
    nama = name_entry.get().strip()
    try:
        latitude = float(lat_entry.get().strip())
        longitude = float(lon_entry.get().strip())
    except ValueError:
        messagebox.showerror("Kesalahan", "Latitude dan Longitude harus berupa angka yang valid.")
        return
    if not nama:
        messagebox.showerror("Kesalahan", "Nama lokasi tidak boleh kosong.")
        return
    if nama in locations:
        messagebox.showerror("Kesalahan", "Lokasi sudah ada.")
        return
    locations[nama] = [latitude, longitude]
    update_location_list()
    name_entry.delete(0, ctk.END)
    lat_entry.delete(0, ctk.END)
    lon_entry.delete(0, ctk.END)

# Update the location list with checkboxes
def update_location_list():
    for widget in location_list_frame.winfo_children():
        widget.destroy()

    global location_checkboxes
    location_checkboxes = {}
    # Tambahkan lokasi tetap
    label = ctk.CTkLabel(location_list_frame, text=f"CV. Sentosa Sumber Mukjizat (-0.024034, 109.339877) [Tetap]")
    label.pack(anchor="w")

    # Tambahkan lokasi lainnya dengan checkbox
    for nama, coords in locations.items():
        if nama == "CV. Sentosa Sumber Mukjizat":
            continue
        var = ctk.BooleanVar()
        checkbox = ctk.CTkCheckBox(location_list_frame, text=f"{nama} ({coords[0]:.6f}, {coords[1]:.6f})", variable=var)
        checkbox.pack(anchor="w")
        location_checkboxes[nama] = var

# Optimize route using Genetic Algorithm
def optimize_route():
    selected_locations = {"CV. Sentosa Sumber Mukjizat": locations["CV. Sentosa Sumber Mukjizat"]}
    for name, var in location_checkboxes.items():
        if var.get():
            selected_locations[name] = locations[name]

    if len(selected_locations) < 2:
        messagebox.showerror("Error", "Please select at least one location in addition to the starting point.")
        return

    distance_matrix, graph, location_nodes = calculate_network_distance_matrix(selected_locations)
    ga = GeneticAlgorithm(selected_locations, distance_matrix)
    best_route_indices, best_distance = ga.optimize()
    optimized_route = [list(selected_locations.keys())[i] for i in best_route_indices]
    route_string = " -> ".join(optimized_route) + f" -> {optimized_route[0]}"
    messagebox.showinfo(
        "Optimized Route",
        f"Rute Optimal:\n{route_string}\n\nTotal Jarak: {best_distance / 1000:.2f} km"
    )
    route_map = plot_route(selected_locations, graph, best_route_indices, location_nodes)
    global map_file  # Make file name accessible for the "Open File Location" button
    map_file = "optimized_route_map.html"
    route_map.save(map_file)
    webbrowser.open(map_file)  # Automatically open the map in the browser
    messagebox.showinfo("Success", f"Optimized route saved as '{map_file}'.")

# Open file location
def open_file_location():
    if os.path.exists(map_file):
        os.startfile(os.path.dirname(os.path.abspath(map_file)))  # Opens file location
    else:
        messagebox.showerror("Error", "The file does not exist.")

# Calculate network distance matrix using OSMnx
def calculate_network_distance_matrix(locations):
    graph = ox.graph_from_point(list(locations.values())[0], dist=5000, network_type='drive')
    location_nodes = {name: ox.distance.nearest_nodes(graph, coord[1], coord[0]) for name, coord in locations.items()}
    n = len(location_nodes)
    distance_matrix = np.zeros((n, n))
    for i, loc1 in enumerate(location_nodes.values()):
        for j, loc2 in enumerate(location_nodes.values()):
            if i != j:
                try:
                    distance_matrix[i, j] = nx.shortest_path_length(graph, loc1, loc2, weight='length')
                except nx.NetworkXNoPath:
                    distance_matrix[i, j] = float('inf')
    return distance_matrix, graph, location_nodes

# Genetic Algorithm class
class GeneticAlgorithm:
    def __init__(self, locations, distance_matrix, population_size=100, generations=500, mutation_rate=0.1):
        self.locations = list(locations.keys())
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.start_end = self.locations.index("CV. Sentosa Sumber Mukjizat")
        self.population = self.initialize_population()

    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            route = np.random.permutation(len(self.locations))
            route = [self.start_end] + [loc for loc in route if loc != self.start_end] + [self.start_end]
            population.append(route)
        return population

    def fitness(self, route):
        total_distance = sum(
            self.distance_matrix[route[i], route[i + 1]] for i in range(len(route) - 1)
        )
        return total_distance if not np.isnan(total_distance) and not np.isinf(total_distance) else float('inf')

    def select_parents(self):
        fitness_scores = [self.fitness(route) for route in self.population]
        probabilities = np.exp(-np.array(fitness_scores))
        if probabilities.sum() == 0 or np.any(np.isnan(probabilities)):
            probabilities = np.ones(len(fitness_scores)) / len(fitness_scores)
        else:
            probabilities /= probabilities.sum()
        parents_indices = np.random.choice(range(self.population_size), size=2, p=probabilities)
        return self.population[parents_indices[0]], self.population[parents_indices[1]]

    def crossover(self, parent1, parent2):
        size = len(parent1)
        start, end = sorted(np.random.choice(range(1, size - 1), size=2, replace=False))
        child = np.full(size, -1)
        child[0], child[-1] = self.start_end, self.start_end
        child[start:end] = parent1[start:end]
        pointer = 1
        for gene in parent2:
            if gene not in child and gene != self.start_end:
                while child[pointer] != -1:
                    pointer += 1
                child[pointer] = gene
        return child

    def mutate(self, route):
        if np.random.rand() < self.mutation_rate:
            i, j = np.random.choice(range(1, len(route) - 1), size=2, replace=False)
            route[i], route[j] = route[j], route[i]
        return route

    def evolve(self):
        new_population = []
        for _ in range(self.population_size):
            parent1, parent2 = self.select_parents()
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        self.population = new_population

    def optimize(self):
        for _ in range(self.generations):
            self.evolve()
        best_route = min(self.population, key=self.fitness)
        return best_route, self.fitness(best_route)

# Plot the route on a map
def plot_route(locations, graph, route, location_nodes):
    route_map = folium.Map(location=list(locations.values())[0], zoom_start=14)
    route_coords = []
    location_names = list(locations.keys())
    for i in range(len(route) - 1):
        loc1 = location_nodes[location_names[route[i]]]
        loc2 = location_nodes[location_names[route[i + 1]]]
        path = nx.shortest_path(graph, loc1, loc2, weight="length")
        path_coords = [(graph.nodes[node]['y'], graph.nodes[node]['x']) for node in path]
        route_coords.extend(path_coords)
    folium.PolyLine(route_coords, color="blue", weight=2.5).add_to(route_map)
    for i, loc_index in enumerate(route):
        name = location_names[loc_index]
        coord = locations[name]
        folium.Marker(coord, tooltip=f"{i + 1}: {name}", icon=folium.Icon(color="green" if i == 0 else "blue")).add_to(route_map)
    return route_map

# Main application window
ctk.set_appearance_mode("Dark")  # Set theme to "Dark" or "Light"
root = ctk.CTk()
root.title("Route Optimization Tool")

# Input frame
input_frame = ctk.CTkFrame(root)
input_frame.grid(row=0, column=0, padx=10, pady=10)

ctk.CTkLabel(input_frame, text="Name:").grid(row=0, column=0, sticky="w")
name_entry = ctk.CTkEntry(input_frame)
name_entry.grid(row=0, column=1)

ctk.CTkLabel(input_frame, text="Latitude:").grid(row=1, column=0, sticky="w")
lat_entry = ctk.CTkEntry(input_frame)
lat_entry.grid(row=1, column=1)

ctk.CTkLabel(input_frame, text="Longitude:").grid(row=2, column=0, sticky="w")
lon_entry = ctk.CTkEntry(input_frame)
lon_entry.grid(row=2, column=1)

ctk.CTkButton(input_frame, text="Add", command=add_location).grid(row=3, column=0, columnspan=2, pady=5)

# Location list frame
list_frame = ctk.CTkFrame(root)
list_frame.grid(row=1, column=0, padx=10, pady=10)

location_list_frame = ctk.CTkFrame(list_frame)
location_list_frame.pack(fill="both", expand=True)

# Action buttons
action_frame = ctk.CTkFrame(root)
action_frame.grid(row=2, column=0, pady=10)

def delete_selected_locations():
    global locations
    selected_to_delete = [nama for nama, var in location_checkboxes.items() if var.get()]
    if not selected_to_delete:
        messagebox.showerror("Kesalahan", "Tidak ada lokasi yang dipilih untuk dihapus.")
        return
    for nama in selected_to_delete:
        if nama in locations:
            del locations[nama]
    update_location_list()
    save_locations_to_json("locations.json")
    messagebox.showinfo("Info", f"Lokasi berhasil dihapus: {', '.join(selected_to_delete)}")
def toggle_mode():
    current_mode = ctk.get_appearance_mode()
    if current_mode == "Dark":
        ctk.set_appearance_mode("Light")  # Ubah ke mode terang
        mode_button.configure(text="Dark Mode")  # Ubah teks tombol
    else:
        ctk.set_appearance_mode("Dark")  # Ubah ke mode gelap
        mode_button.configure(text="Light Mode")  # Ubah teks tombol

mode_button = ctk.CTkButton(action_frame, text="Light Mode", command=toggle_mode)
mode_button.pack(side="left", padx=5)
# Tombol untuk menghapus lokasi terpilih
delete_button = ctk.CTkButton(action_frame, text="Hapus Lokasi Terpilih", command=delete_selected_locations)
delete_button.pack(side="left", padx=5)


ctk.CTkButton(action_frame, text="Optimasi Rute", command=optimize_route).pack(side="left", padx=5)
ctk.CTkButton(action_frame, text="Simpan Lokasi Ke Database", command=lambda: save_locations_to_json("locations.json")).pack(side="left", padx=5)

# Load initial locations
locations = load_locations_from_json("locations.json")
update_location_list()

root.mainloop()