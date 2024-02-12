from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import threading
import time

app = Flask(__name__)

# Fonctions pour l'algorithme de clustering de pics de densité
def calculate_distances(thread_id, num_threads, points, distances, thread_distances):
    num_points = points.shape[0]
    chunk_size = num_points // num_threads
    start = thread_id * chunk_size
    end = start + chunk_size if thread_id < num_threads - 1 else num_points
    
    for i in range(start, end):
        for j in range(num_points):
            distances[i][j] = np.linalg.norm(points[i] - points[j])
            thread_distances[i][j] = thread_id

def calculate_local_density(thread_id, num_threads, distances, local_densities, cutoff_distance):
    num_points = distances.shape[0]
    chunk_size = num_points // num_threads
    start = thread_id * chunk_size
    end = start + chunk_size if thread_id < num_threads - 1 else num_points
    
    for i in range(start, end):
        local_density = sum(1 for dist in distances[i] if dist < cutoff_distance)
        local_densities[i] = local_density

def find_cluster_centers(local_densities, distances, num_clusters):
    sorted_indices = np.argsort(local_densities)[::-1]
    cluster_centers = []

    for i in range(num_clusters):
        cluster_centers.append(sorted_indices[i])

    return cluster_centers

def assign_clusters(distances, cluster_centers, num_clusters):
    num_points = distances.shape[0]
    cluster_assignments = np.zeros(num_points, dtype=int)

    for i in range(num_points):
        closest_center = np.argmin([distances[i][center] for center in cluster_centers])
        cluster_assignments[i] = closest_center

    return cluster_assignments

def density_peaks_clustering(points, cutoff_distance=1.0, num_clusters=3):
    num_points = points.shape[0]
    distances = np.zeros((num_points, num_points))
    thread_distances = np.zeros((num_points, num_points))
    local_densities = np.zeros(num_points)
    
    num_threads = 4
    threads = []
    for i in range(num_threads):
        thread = threading.Thread(target=calculate_distances, args=(i, num_threads, points, distances, thread_distances))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()
    
    threads.clear()
    for i in range(num_threads):
        thread = threading.Thread(target=calculate_local_density, args=(i, num_threads, distances, local_densities, cutoff_distance))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()

    # Find cluster centers
    cluster_centers = find_cluster_centers(local_densities, distances, num_clusters)

    # Assign clusters
    cluster_assignments = assign_clusters(distances, cluster_centers, num_clusters)

    # Display clustering results
    print("Density Peaks Clustering Results:")
    for i in range(num_points):
        print(f"Point {i + 1}: Density = {local_densities[i]}, Calculated by Thread {int(thread_distances[i][0]) + 1}, Cluster = {cluster_assignments[i]}")

    return cluster_assignments

# Routes Flask
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file:
            # Charger le dataset
            data = pd.read_csv(file)

            # Sélectionner les colonnes pour le clustering
            X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]

            # Normaliser les données
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Exécuter l'algorithme de clustering
            start_time = time.time()
            cluster_labels = density_peaks_clustering(X_scaled, num_clusters=3)
            end_time = time.time()
            execution_time = end_time - start_time

            # Ajouter les labels au dataframe d'origine
            data['cluster_label'] = cluster_labels

            # Informations sur le dataset
            dataset_info = {
                'taille': len(data),
                'colonnes': list(data.columns)
            }

            # Résultats d'évaluations d'algorithme de clustering
            clustering_results = {
                'temps_execution': execution_time,
                'nombre_clusters': len(np.unique(cluster_labels))
            }

            return render_template('resultats.html', data=data.to_html(), dataset_info=dataset_info, clustering_results=clustering_results)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
