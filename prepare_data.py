import os
import zipfile
import shutil
import urllib.request
import ssl

# Bypass SSL certificate verification for downloads (common macOS Python issue)
ssl._create_default_https_context = ssl._create_unverified_context

import sys

def show_progress(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = min(100, read_so_far * 100 / total_size)
        sys.stdout.write(f"\rDownloading... {percent:.2f}% ({read_so_far / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB)")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rDownloading... {read_so_far / (1024*1024):.2f} MB")
        sys.stdout.flush()

def download_file(url, filepath):
    try:
        req = urllib.request.Request(url, method='HEAD')
        with urllib.request.urlopen(req) as resp:
            content_length = resp.getheader('Content-Length')
            expected_size = int(content_length) if content_length else None
    except Exception:
        expected_size = None

    if os.path.exists(filepath):
        if expected_size and os.path.getsize(filepath) == expected_size:
            print(f"File {filepath} already exists and is complete. Skipping download.")
            return
        elif not expected_size and os.path.getsize(filepath) > 1024*1024*50:
            print(f"File {filepath} already exists and seems complete (>50MB). Skipping download.")
            return
        else:
            print(f"Local file {filepath} is incomplete or corrupted. Deleting and re-downloading...")
            os.remove(filepath)

    print(f"Downloading {url} to {filepath}...")
    urllib.request.urlretrieve(url, filepath, reporthook=show_progress)
    print("\nDownload completed.")

def unzip_file(zip_filepath, dest_dir):
    print(f"Unzipping {zip_filepath} to {dest_dir}...")
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    print("Unzip completed.")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    urls = {
        "GTSRB_Final_Training_Images.zip": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip",
        "GTSRB_Final_Test_Images.zip": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip",
        "GTSRB_Final_Test_GT.zip": "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip"
    }

    # Step 1: Download files
    for filename, url in urls.items():
        filepath = os.path.join(data_dir, filename)
        download_file(url, filepath)

    # Step 2: Unzip files
    for filename in urls.keys():
        filepath = os.path.join(data_dir, filename)
        unzip_file(filepath, data_dir)

    # Step 3: Structure test set
    gtsrb_dir = os.path.join(data_dir, 'GTSRB')
    images_dir = os.path.join(gtsrb_dir, 'Final_Test/Images')
    test_dir = os.path.join(gtsrb_dir, 'test')
    os.makedirs(test_dir, exist_ok=True)

    csv_path = os.path.join(data_dir, 'GT-final_test.csv')
    if not os.path.exists(csv_path):
        # check if it is inside another folder or extracted directly
        # sometimes GT-final_test.csv is inside the zip or extracted directly to data
        pass

    print("Structuring test set classes...")
    with open(csv_path, 'r') as f:
        image_names = f.readlines()

    count = 0
    for text in image_names[1:]:
        parts = text.strip().split(';')
        if len(parts) < 8:
            continue
        image_name = parts[0]
        class_id = int(parts[-1])

        test_class_dir = os.path.join(test_dir, f"{class_id:04d}")
        os.makedirs(test_class_dir, exist_ok=True)
        
        src_image_path = os.path.join(images_dir, image_name)
        dst_image_path = os.path.join(test_class_dir, image_name)
        
        shutil.copy(src_image_path, dst_image_path)
        count += 1

    print(f"Data preparation complete! Copied {count} images to structured test directory.")

if __name__ == '__main__':
    main()
