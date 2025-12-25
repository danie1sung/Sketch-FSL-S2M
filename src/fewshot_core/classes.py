# Fixed list of 13 classes for the few-shot learning task.
CLASSES = [
  "Airplane", "Bench", "Cabinet", "Car", "Chair", "Display",
  "Lamp", "Loudspeaker", "Rifle", "Sofa", "Table", "Telephone", "Watercraft"
]

# Mapping from class name to a unique integer ID.
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# Mapping from ID back to class name.
ID_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}
