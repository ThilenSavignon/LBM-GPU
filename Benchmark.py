import subprocess
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import de la barre de progression
import re  # Pour extraire le temps

executable = "./main"
labels = ["Shared", "Global"]
resolutions = [(256, 256), (1024, 1024)]
iterations = [3000 * i for i in range(1, 11)]

total_tasks = (len(resolutions) * len(iterations)) * 2
results = {label: {} for label in labels}

with tqdm(total=total_tasks, desc="Exécution", unit="task") as pbar:
    for label in labels:
        for w, h in resolutions:
            times = []
            for i in iterations:
                if label == "Shared":
                    cmd = [executable, f"--w={w}", f"--h={h}", f"--i={i}", "--s"]
                else:
                    cmd = [executable, f"--w={w}", f"--h={h}", f"--i={i}"]
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Extraire la valeur de temps depuis l'avant-dernière ligne
                output_lines = result.stdout.strip().split('\n')
                time_line = output_lines[-2]  # Avant-dernière ligne
                
                # Extraire le temps en millisecondes avec une expression régulière
                last_line = result.stdout.strip().split('\n')[-1]
                match = re.search(r'Elapsed time:\s*([\d.]+)\s*ms', last_line)
                if match:
                    exec_time = float(match.group(1)) / 1000  # Convertir en secondes
                    times.append(exec_time)
                else:
                    raise ValueError("Impossible de trouver le temps dans la sortie.")
                
                pbar.update(1)

            results[label][f"{w}x{h}"] = times

plt.figure(figsize=(12, 7))
for label, data in results.items():
    for res, times in data.items():
        plt.plot(iterations, times, marker='o', label=f"{label} - {res}")

plt.xlabel('Iterations')
plt.ylabel('Time (s)')
plt.legend()
plt.grid(True)
plt.show()
