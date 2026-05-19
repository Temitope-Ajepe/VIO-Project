import csv

results = [
    # Sequence, Method, Alignment, ATE, RPE, Runtime, Drift
    ['Room2',     'VO',  'Sim3', 0.7224, 0.0150, 137.00, 'Full GT'],
    ['Room2',     'VIO', 'SE3',  0.7564, 0.9995, 'N/A',  'Full GT'],
    ['Corridor3', 'VO',  'Sim3', 1.1316, 0.0107, 75.11,  321.81],
    ['Corridor3', 'VIO', 'SE3',  1.0545, 0.9996, 'N/A',  'N/A'],
    ['Outdoors5', 'VO',  'Sim3', 1.0876, 0.0107, 129.93, 978.22],
    ['Outdoors5', 'VIO', 'SE3',  0.8673, 1.0004, 'N/A',  'N/A'],
]

with open('results/tables/results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'Sequence', 'Method', 'Alignment', 
        'ATE (m)', 'RPE (m/100m)', 
        'Runtime (ms/frame)', 'Drift (m)'
    ])
    writer.writerows(results)

print("Results table saved to results/tables/results.csv")