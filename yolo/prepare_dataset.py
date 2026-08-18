"""Подготовка датасета в формате YOLO."""
from pathlib import Path
import shutil
import yaml
from tqdm import tqdm


def copy_label_with_optional_shift(src_label_path, dst_label_path, class_shift=0):
    """Копирует YOLO-метку с опциональным сдвигом class_id."""
    dst_label_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not src_label_path.exists():
        dst_label_path.write_text("")
        return
    
    new_lines = []
    with open(src_label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"⚠ Некорректная строка в {src_label_path}: {line}")
                continue
            
            class_id = int(float(parts[0])) - class_shift
            if class_id < 0:
                raise ValueError(f"Отрицательный class_id={class_id} в {src_label_path}")
            
            coords_float = list(map(float, parts[1:]))
            if not all(0.0 <= x <= 1.0 for x in coords_float):
                print(f"⚠ Координаты вне [0,1] в {src_label_path}")
            
            new_lines.append(f"{class_id} {' '.join(parts[1:])}\n")
    
    with open(dst_label_path, "w") as f:
        f.writelines(new_lines)


def prepare_yolo_split(src_images_dir, src_labels_dir,
                       dst_images_dir, dst_labels_dir,
                       image_ext=".png", class_shift=0):
    """Подготавливает train или val для YOLO."""
    src_images_dir = Path(src_images_dir)
    src_labels_dir = Path(src_labels_dir)
    dst_images_dir = Path(dst_images_dir)
    dst_labels_dir = Path(dst_labels_dir)
    
    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = sorted(src_images_dir.glob(f"*{image_ext}"))
    if len(image_paths) == 0:
        raise RuntimeError(f"Не найдено изображений в: {src_images_dir}")
    
    print(f"Найдено {len(image_paths)} изображений в {src_images_dir}")
    
    for img_path in tqdm(image_paths, desc=f"Подготовка {dst_images_dir.name}"):
        shutil.copy2(img_path, dst_images_dir / img_path.name)
        
        src_label = src_labels_dir / f"{img_path.stem}.txt"
        dst_label = dst_labels_dir / f"{img_path.stem}.txt"
        copy_label_with_optional_shift(src_label, dst_label, class_shift)


def create_data_yaml(dataset_root, save_path):
    """Сканирует классы и создаёт data.yaml для YOLO."""
    dataset_root = Path(dataset_root)
    
    class_ids = set()
    for labels_dir in [dataset_root / "labels" / "train", dataset_root / "labels" / "val"]:
        for txt_path in labels_dir.glob("*.txt"):
            with open(txt_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_ids.add(int(float(parts[0])))
    
    class_ids = sorted(class_ids)
    print("Найденные class_id:", class_ids)
    
    if len(class_ids) == 0:
        nc = 1
        names = ["text"]
    else:
        nc = max(class_ids) + 1
        names = ["text"] if nc == 1 else [f"class_{i}" for i in range(nc)]
    
    data_yaml = {
        "path": str(dataset_root),
        "train": "images/train",
        "val": "images/val",
        "nc": nc,
        "names": names,
    }
    
    with open(save_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, allow_unicode=True)
    
    print(f"✓ data.yaml сохранён: {save_path}")
    return data_yaml
