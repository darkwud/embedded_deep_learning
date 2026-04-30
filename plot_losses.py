import csv
import matplotlib.pyplot as plt

# Configuration: which CSV files to plot, and how each should appear
runs = [
    {'file': 'loss_log_lr_0.1.csv',     'label': 'lr = 0.1',      'color': 'red'},
    {'file': 'loss_log_lr_0.001.csv',   'label': 'lr = 0.001',    'color': 'blue'},
    {'file': 'loss_log_lr_0.00001.csv',  'label': 'lr = 0.00001',  'color': 'green'},
]

# Create a figure with two subplots side-by-side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Read each CSV and plot both curves
for run in runs:
    epochs = []
    losses = []
    accuracies = []
    with open(run['file'], 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row['epoch']))
            losses.append(float(row['train_loss']))
            accuracies.append(float(row['val_accuracy']))

    # Left subplot: training loss
    ax1.plot(epochs, losses,
             marker='o', linewidth=2,
             color=run['color'], label=run['label'])

    # Right subplot: validation accuracy
    ax2.plot(epochs, accuracies,
             marker='o', linewidth=2,
             color=run['color'], label=run['label'])

# Style the loss subplot
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Training Loss', fontsize=12)
ax1.set_title('Training Loss vs. Epoch', fontsize=14)
ax1.set_yscale('log')                                       # log scale shows all curves clearly
ax1.legend(fontsize=11)
ax1.grid(True, which='both', linestyle='--', alpha=0.5)

# Style the accuracy subplot
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Validation Accuracy', fontsize=12)
ax2.set_title('Validation Accuracy vs. Epoch', fontsize=14)
ax2.set_ylim(0, 1)                                          # accuracy is bounded [0, 1]
ax2.legend(fontsize=11, loc='lower right')
ax2.grid(True, linestyle='--', alpha=0.5)

# Overall title for the figure
fig.suptitle('Effect of Learning Rate on ResNet-18 Training', fontsize=16, y=1.02)

plt.tight_layout()

# Save and display
plt.savefig('loss_and_accuracy.png', dpi=100, bbox_inches='tight')
plt.show()