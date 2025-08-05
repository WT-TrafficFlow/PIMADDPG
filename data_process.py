import pandas as pd
import numpy as np
import os
# Load the CSV file
data = pd.read_csv('sumo_data.csv')

# Define constants
rows_per_segment = 240
rows_to_remove = 10
window_size = 13
step_size = 1

# Split the data into segments of 240 rows each
num_segments = len(data) // rows_per_segment
segments = [
    data.iloc[i * rows_per_segment:(i + 1) * rows_per_segment].reset_index(drop=True)
    for i in range(num_segments)
]

# Process each segment
processed_segments = []
for segment in segments:
    # Remove the first 10 rows
    trimmed_segment = segment.iloc[rows_to_remove:]

    # Convert to numpy array for sliding window slicing
    trimmed_array = trimmed_segment.to_numpy()

    # Create sliding windows
    num_samples = (len(trimmed_array) - window_size) // step_size + 1
    windows = np.array([
        trimmed_array[i:i + window_size] for i in range(0, num_samples, step_size)
    ])

    processed_segments.append(windows)

# Stack all segments into a 3D array
final_array = np.concatenate(processed_segments, axis=0)
print(f"Processed data saved as 'processed_simulation.npz' with shape {final_array.shape}")
np.random.shuffle(final_array)
total_samples = final_array.shape[0]
# train_end = int(0.7 * total_samples)
# # Normalize each feature
# means = np.mean(final_array[:train_end, ...], axis=(0, 1))
# stds = np.std(final_array[:train_end, ...], axis=(0, 1))
#
# # Normalize in chunks
# chunk_size = 1000  # Adjust based on memory constraints
# normalized_array = []
# for i in range(0, final_array.shape[0], chunk_size):
#     chunk = final_array[i:i + chunk_size]
#     normalized_chunk = (chunk - means) / stds
#     normalized_array.append(normalized_chunk)
#
# normalized_array = np.concatenate(normalized_array, axis=0)
normalized_array = final_array


# Save the 3D array to an NPZ file


# 假设你的原始数组是 normalized_array，形状为 (N, 13, 40)
N = normalized_array.shape[0]

# 1. 提取前36个特征并重塑为 (N, 13, 12, 3)
part1 = normalized_array[:, :, :36].reshape(N, window_size, 12, 3)
# 2. 提取最后4个特征并扩展为 (N, 13, 4, 1)
last_four = normalized_array[:, :, 36:40].reshape(N, window_size, 4, 1)
last_four2 = normalized_array[:, :, 40:].reshape(N, window_size, 4, 1)
# 3. 沿最后一维重复 3 次，得到形状 (N, 13, 12, 1)
part2 = np.tile(last_four, (1, 1, 1, 3))
part2 = part2.reshape(N, window_size, 1, 12).transpose(0, 1, 3, 2)
part3 = np.tile(last_four2, (1, 1, 1, 3))
part3 = part3.reshape(N, window_size, 1, 12).transpose(0, 1, 3, 2)
final_array = np.concatenate([part1, part2, part3], axis=-1)
x = final_array
y = final_array[:, -1:, :, :2]

print("x shape: ", x.shape, ", y shape: ", y.shape)
# Write the data into npz file.
num_samples = x.shape[0]
num_test = round(num_samples * 0.2)
num_train = round(num_samples * 0.7)
num_val = num_samples - num_test - num_train


# train
x_train, y_train = x[:num_train], y[:num_train]
# val
x_val, y_val = (
    x[num_train: num_train + num_val],
    y[num_train: num_train + num_val],
)
# test
x_test, y_test = x[-num_test:], y[-num_test:]

for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join("%s.npz" % cat),
            x=_x,
            y=_y)


