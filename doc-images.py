## Copyright © 2023, Alex J. Champandard.  Licensed under MIT; see LICENSE! ⚘
"""
This script takes a folder with _image_ training data, as a set of TAR files, and creates
documentation for each of the items inside. The result is a JSON manifest file that can be
used as a summary of the training data, which can be required for regulatory compliance but
also (when signed cryptographically) provides forensic evidence of data sourcing practices
used during ML development.

USAGE:

1) To integrate into your own MLOps pipeline (recommended), copy process_image() and its
two support functions of about 100 lines of code.  Then save the outputs to a JSON file!

2) To run this script standalone, call it by passing a directory that contains .TAR files
as output by `img2dataset`; they'll be scanned in multiple processes an JSON file output.

"""

import io
import time
import warnings
from urllib.parse import urlparse

import ftfy
import iscc
import PIL.Image


ISCC_IMAGE_THUMB_SIZE = (32, 32)
JPEG_IMAGE_MINIMUM_BYTES = 141


def process_image(data: bytes, title: str, url: str, timestamp: int = None) -> dict:
    """
    Extract all relevant information from an image given the data loaded from disk using
    PIL, and return it in dictionary format.
    """

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            img = PIL.Image.open(io.BytesIO(data), formats=None)
            mime_type = get_mime_type(img)
            img = img.convert("L").resize(ISCC_IMAGE_THUMB_SIZE, PIL.Image.LANCZOS)

    except (PIL.UnidentifiedImageError, OSError):
        return {}

    meta_id, _, _ = iscc.meta_id(title, "") if title else ""
    content_id = iscc.content_id_image(img)
    data_id = iscc.data_id(data)
    instance_id, checksum = iscc.instance_id(data)

    domain = urlparse(url).netloc
    doc = {
        "domain": domain,
        "iscc": "-".join([meta_id, content_id, data_id, instance_id]),
        "timestamp": timestamp or int(time.time()),
        "bytes": len(data),
        "checksum": checksum,
        "mime-type": mime_type,
    }

    if (cmi := get_copyright(img)) is not None:
        doc["copyright"] = cmi

    return doc


def get_mime_type(img: PIL.Image) -> str:
    """
    Guess the mime-type of a PIL image based on its format.  Must be called before converting
    or resizing, otherwise the original format is lost.
    """
    return {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "GIF": "image/gif",
        None: "¡ERROR!",
    }.get(img.format, "other/" + img.format)


def get_copyright(img: PIL.Image) -> str:
    """
    Find the Copyright information within a PIL image by looking at EXIF headers.  This function
    checks for the Copyright and Artist fields, rejecting them if they are too small.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            exif_data = img.getexif() or {}
    except SyntaxError:
        return None

    for tag in [33432, 315]:  # Copyright, Artist
        if tag not in exif_data:
            continue
        value = str(exif_data[tag]).strip()
        if len(value) < 2 or "[None]" in value:
            continue
        return ftfy.fix_text(value)

    return None


########################################################################################
"""
Standalone doc-images.py script starts here; run with a directory as argument.
"""

import os
import json
import tarfile
import argparse
import multiprocessing

from tqdm import tqdm


def process_tar(filename: str) -> list[str]:
    """
    Scan and process each of the files within a TAR file in order to extract their
    properties, and then return a list of strings containing JSON for each item.
    """

    if os.path.isfile(filename.replace(".tar", "_doc.jsonl")):
        return []

    pid = multiprocessing.current_process()._identity[0]
    lines = []
    cache = {}

    with tarfile.open(filename, "r") as tar:
        for member in tqdm(
            tar.getmembers(),
            desc=filename.split("/")[-1],
            position=pid + 1,
            mininterval=1.0,
        ):
            member_name = member.name.lower()
            key = member_name.rsplit(".", maxsplit=2)[0]

            if member_name.endswith(".jpg"):
                file = tar.extractfile(member)
                cache[key] = file.read()
                del file

            if member_name.endswith(".json"):
                assert key in cache
                data = cache.pop(key)

                meta = json.load(tar.extractfile(member))
                doc = process_image(
                    data,
                    title=meta.get('caption', ''),
                    url=meta["url"],
                    timestamp=int(member.mtime),
                )

                if len(doc) > 0:
                    lines.append(json.dumps(doc))

    with open(filename.replace(".tar", "_doc.jsonl"), "w") as doc:
        doc.writelines(l + "\n" for l in lines)

    return lines


def main(path, out_file: str = None, num_proc=4):
    """
    Create the documentation for a directory with image/text data stored in TAR files,
    then save it to a JSON file with per-item breakdown.
    """

    path = os.path.abspath(path)
    filenames = sorted(
        [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".tar")]
    )
    assert len(filenames) > 0, f"No TAR files found in {path}."

    with multiprocessing.Pool(num_proc) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(process_tar, filenames),
                total=len(filenames),
                desc=os.path.normpath(path).split("/")[-1],
                position=0,
            )
        )

    out_file = out_file or os.path.normpath(path) + ".json"
    with open(out_file, "wb") as f:
        f.write(b"[\n")
        for lines in results:
            f.writelines([f"  {ln},\n".encode() for ln in lines])

        f.seek(-2, os.SEEK_CUR)
        f.write(b"\n]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, help='Training data stored as TAR files containing .jpg and .json entries.')
    args = parser.parse_args()

    main(args.directory)
