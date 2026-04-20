import os
import yaml

def check_yaml(yaml_path):
    print("\n🔍 Checking YAML file...")
    
    if not os.path.exists(yaml_path):
        print(f"❌ YAML not found: {yaml_path}")
        return None
    
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    required_keys = ["path", "train", "val", "nc", "names"]
    for key in required_keys:
        if key not in data:
            print(f"❌ Missing key in YAML: {key}")
            return None
    
    print("✅ YAML structure is valid")
    return data


def check_paths(base_path, split):
    img_path = os.path.join(base_path, "images", split)
    lbl_path = os.path.join(base_path, "labels", split)

    print(f"\n🔍 Checking {split} paths...")
    print(f"Images: {img_path}")
    print(f"Labels: {lbl_path}")

    if not os.path.exists(img_path):
        print(f"❌ Missing images folder: {img_path}")
        return False
    if not os.path.exists(lbl_path):
        print(f"❌ Missing labels folder: {lbl_path}")
        return False

    images = [f for f in os.listdir(img_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    labels = [f for f in os.listdir(lbl_path) if f.endswith('.txt')]

    print(f"✅ Found {len(images)} images, {len(labels)} labels")

    if len(images) == 0:
        print("❌ No images found")
        return False

    return img_path, lbl_path, images, labels


def check_labels(img_path, lbl_path, images, nc):
    print("\n🔍 Checking label consistency...")

    errors = 0

    for img in images:
        name = os.path.splitext(img)[0]
        label_file = os.path.join(lbl_path, name + ".txt")

        if not os.path.exists(label_file):
            print(f"❌ Missing label for: {img}")
            errors += 1
            continue

        with open(label_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            
            if len(parts) != 5:
                print(f"❌ Invalid format in {label_file}: {line}")
                errors += 1
                continue

            cls = int(parts[0])
            if cls >= nc:
                print(f"❌ Invalid class index in {label_file}: {cls}")
                errors += 1

    if errors == 0:
        print("✅ All labels are valid")
    else:
        print(f"⚠️ Found {errors} issues in labels")


def main():
    yaml_path = "dataset/vivarium.yaml"

    data = check_yaml(yaml_path)
    if not data:
        return

    base_path = data["path"]
    nc = data["nc"]

    train = check_paths(base_path, "train")
    val = check_paths(base_path, "val")

    if train:
        check_labels(train[0], train[1], train[2], nc)

    if val:
        check_labels(val[0], val[1], val[2], nc)

    print("\n🎯 Verification completed!")


if __name__ == "__main__":
    main()