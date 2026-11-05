import csv
import io
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import fire
import requests
from PIL import Image


def download(url: str, output_file: Path):
    if output_file.exists():
        return f"Skipped {output_file} (already exists)."
    try:
        buffer = tempfile.SpooledTemporaryFile(max_size=1_000_000)  # noqa
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=1024):
                downloaded += len(chunk)
                buffer.write(chunk)
            buffer.seek(0)
            img = Image.open(io.BytesIO(buffer.read()))
            img.save(output_file)
        buffer.close()
        return f"Downloaded {output_file}."
    except Exception as e:  # noqa
        return f"Failed to download {output_file}: {e}."


def main(csv_file: str | Path = "RAISE_1k.csv", output_folder: str | Path = "data/raise1k"):
    filenames, urls = [], []
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            filenames.append(row[0])
            urls.append(row[2])

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    count = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(download, url, (output_folder / filename).with_suffix(".png"))
            for (url, filename) in zip(urls, filenames)
        ]
        for future in as_completed(futures):
            count += 1
            print(f"{count:>4} / {len(futures)}: {future.result()}")


if __name__ == "__main__":
    fire.Fire(main)
