import random

OLD_INDICES_FILE = "eval_indices_300.txt"
OUTPUT_FILE = "test_indices_300.txt"

DATASET_SIZE = 5369
NUM_SAMPLES = 300
SEED = 42


# Load old indices
with open(OLD_INDICES_FILE, "r") as f:
    old_indices = {
        int(line.strip())
        for line in f
        if line.strip()
    }

# All indices not used in the old evaluation set
available_indices = [
    i for i in range(DATASET_SIZE)
    if i not in old_indices
]

if len(available_indices) < NUM_SAMPLES:
    raise ValueError(
        f"Only {len(available_indices)} unused indices available, "
        f"but {NUM_SAMPLES} were requested."
    )

# Reproducible random sample
random.seed(SEED)
test_indices = random.sample(
    available_indices,
    NUM_SAMPLES
)

# Sorting isn't required, but makes the file easier to inspect
test_indices.sort()

with open(OUTPUT_FILE, "w") as f:
    for idx in test_indices:
        f.write(f"{idx}\n")

print(f"Old indices: {len(old_indices)}")
print(f"Generated new indices: {len(test_indices)}")
print(f"Overlap: {len(set(test_indices) & old_indices)}")
print(f"Saved to: {OUTPUT_FILE}")