from pathlib import Path
from PIL import Image

DATA_DIR = Path("data/chest_xray")

splits = ["train", "val", "test"]
classes = ["NORMAL", "PNEUMONIA"]

total = 0

for split in splits:
    print(f"\n[{split.upper()}]")

    for cls in classes:
        folder = DATA_DIR / split / cls

        files = [
            file for file in folder.iterdir()
            if file.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        corrupted = []

        for file in files:
            try:
                with Image.open(file) as img:
                    img.verify()
            except Exception:
                corrupted.append(file)

        print(
            f"{cls:10s}: {len(files):4d} files | "
            f"{len(corrupted):3d} corrupted"
        )

        total += len(files)

print(f"\nTotal files: {total}")