import os
import requests
import zipfile

# The URL of the zip file

# Get the directory of the current script


def load_from_AISquare(url, dir):
    dir_path = os.path.dirname(os.path.realpath(__file__))

    os.makedirs(os.path.join(dir_path, str(dir)), exist_ok=True)

    # Download the file
    response = requests.get(url)
    print(os.path.join(dir_path, str(dir), "data.zip"))
    with open(os.path.join(dir_path, str(dir), "data.zip"), "wb") as file:
        file.write(response.content)

    # Unzip the file into str(dir) directory
    with zipfile.ZipFile(os.path.join(dir_path, str(dir), "data.zip"), "r") as zip_ref:
        zip_ref.extractall(os.path.join(dir_path, str(dir)))

    # Delete the zip file
    os.remove(os.path.join(dir_path, str(dir),"data.zip"))


if __name__ == "__main__":
    urls = ["https://aisquare-zjk.oss-cn-zhangjiakou.aliyuncs.com/AIS-Square/datasets/H2O-PBE0TS/PBE0-TS-H2O.zip",
            "https://aisquare-zjk.oss-cn-zhangjiakou.aliyuncs.com/AIS-Square/datasets/H2O-Phase-Diagram/H2O-Phase-Diagram.zip"]

    load_from_AISquare(urls[0], "H2O-PBE0TS")
    load_from_AISquare(urls[1], "H2O-Phase-Diagram")
    #load_from_AISquare(urls[0], "H2O-PBE0TS")

