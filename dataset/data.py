import subprocess
import os


def download_udacity_dataset():
    home_dir = os.path.expanduser("~")
    data_dir = os.path.join(home_dir, "facial-keypoint-detection", "data")
    os.makedirs(data_dir, exist_ok=True)

    training_dir = os.path.join(data_dir, "training")
    if os.path.exists(training_dir):
        print("Dataset already exists. Skipping download.")
        return

    print("Downloading Udacity Facial Keypoints Dataset...")
    zip_url = "https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aea1b91_train-test-data/train-test-data.zip"
    zip_path = os.path.join(data_dir, "train-test-data.zip")

    if os.name == 'nt':  # Windows
        ps_script = f"""
        Invoke-WebRequest -Uri "{zip_url}" -OutFile "{zip_path}"
        Expand-Archive -Path "{zip_path}" -DestinationPath "{data_dir}" -Force
        Remove-Item "{zip_path}"
        """
        subprocess.run(["powershell", "-Command", ps_script], check=True)
    else:  # Linux / Mac
        bash_script = f"""
        cd "{data_dir}"
        wget -q --show-progress "{zip_url}" -O train-test-data.zip
        unzip -q train-test-data.zip
        rm train-test-data.zip
        """
        subprocess.run(["bash", "-c", bash_script], check=True)

    print(f"✅ Dataset downloaded and extracted to {data_dir}")