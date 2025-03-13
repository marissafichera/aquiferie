import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

# Load the data from the CSV file
data = pd.read_csv('ranking_from_insights.csv')

# Set up the figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Define the categories and the basin columns
categories = data['category'][:-1]
basins = ['TularosaBasin', 'RoswellArtesianBasin', 'AlbuquerqueBasin']

# Create a color map for the different statuses
status_map = {'yes': 'blue', 'no': 'none', 'partial': 'white'}

# Loop through each cell in the matrix
for i, category in enumerate(categories):
    for j, basin in enumerate(basins):
        status = data.at[i, basin]
        print(f'{basin=}; {category=}; {status=};')

        # Create a rectangle for each cell, adjusted by 0.5 to center it
        rect_x = j  # x position of the rectangle (corresponds to basin)
        rect_y = i  # y position of the rectangle (corresponds to category)

        if status == 'yes':
            rect = patches.Rectangle((rect_x - 0.5, rect_y - 0.5), 1, 1, linewidth=1, edgecolor='black',
                                     facecolor='blue')
        elif status == 'no':
            rect = patches.Rectangle((rect_x - 0.5, rect_y - 0.5), 1, 1, linewidth=1, edgecolor='black',
                                     facecolor='none')
        elif status == 'partial':
            rect = patches.Rectangle((rect_x - 0.5, rect_y - 0.5), 1, 1, linewidth=1, edgecolor='blue',
                                     facecolor='none', hatch='/')

        ax.add_patch(rect)
        print(f'{status=}; {ax=}')

# Set the labels for the axes
ax.set_xticks(range(len(basins)))
ax.set_xticklabels(basins)
ax.set_yticks(range(len(categories)))
ax.set_yticklabels(categories)

# Labeling the axes
ax.set_xlabel('Basins')
ax.set_ylabel('Categories')

# Set gridlines to help visualize the matrix
ax.set_xlim(-0.5, len(basins) - 0.5)
ax.set_ylim(-0.5, len(categories) - 0.5)
ax.set_xticks(range(len(basins)), minor=True)
ax.set_yticks(range(len(categories)), minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

# Create legend
legend_elements = [
    patches.Patch(edgecolor='black', facecolor='blue', label='Characterized'),
    patches.Patch(edgecolor='black', facecolor='none', label='Not Characterized'),
    patches.Patch(edgecolor='blue', facecolor='none', hatch='///', label='Partially Characterized')
]

# Adding the legend below the x-axis
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=12)

# Display the plot

# Display the plot
plt.gca().invert_yaxis()  # To match the matrix orientation
plt.tight_layout()
plt.show()