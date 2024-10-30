import argparse
import time
from pathlib import Path
import multiprocessing as mp
from functools import partial
import os

try:
    from lv_img_converter import Converter
except ImportError:
    print(
        """Please install Requirements by using `pip install -r requirements.txt`
Read README.md for more details.
          """
    )
    exit(-1)

name2const = {
    "RGB332": Converter.FLAG.CF_TRUE_COLOR_332,
    "RGB565": Converter.FLAG.CF_TRUE_COLOR_565,
    "RGB565SWAP": Converter.FLAG.CF_TRUE_COLOR_565_SWAP,
    "RGB888": Converter.FLAG.CF_TRUE_COLOR_888,
    "alpha_1": Converter.FLAG.CF_ALPHA_1_BIT,
    "alpha_2": Converter.FLAG.CF_ALPHA_2_BIT,
    "alpha_4": Converter.FLAG.CF_ALPHA_4_BIT,
    "alpha_8": Converter.FLAG.CF_ALPHA_8_BIT,
    "indexed_1": Converter.FLAG.CF_INDEXED_1_BIT,
    "indexed_2": Converter.FLAG.CF_INDEXED_2_BIT,
    "indexed_4": Converter.FLAG.CF_INDEXED_4_BIT,
    "indexed_8": Converter.FLAG.CF_INDEXED_8_BIT,
    "raw": Converter.FLAG.CF_RAW,
    "raw_alpha": Converter.FLAG.CF_RAW_ALPHA,
    "raw_chroma": Converter.FLAG.CF_RAW_CHROMA,
    "true_color": Converter.FLAG.CF_TRUE_COLOR,
    "true_color_alpha": Converter.FLAG.CF_TRUE_COLOR_ALPHA,
    "true_color_chroma": Converter.FLAG.CF_TRUE_COLOR_CHROMA,
}

def check_allowed(filepath: Path):
    suffix: str = filepath.suffix
    return suffix.lower() in [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tga",
        ".gif",
        ".bin",
    ]

def get_dest_path(root_path: Path, filepath: Path, out_path: Path, name: str, ff: str) -> Path:
    """Calculate destination path for converted file"""
    rel_path = Path()
    try:
        rel_path = filepath.relative_to(root_path).parent
    except ValueError:
        rel_path = root_path.relative_to(root_path.anchor)

    out_path = root_path if out_path == Path() else out_path
    out_path = out_path.joinpath(rel_path)

    file_conf = {
        "C": {"suffix": ".c"},
        "BIN": {"suffix": ".bin"},
    }

    return out_path.joinpath(name).with_suffix(file_conf[ff]["suffix"])

def should_convert(src_path: Path, dest_path: Path, min_size: int = 100, force: bool = False) -> bool:
    """
    Determine if conversion is needed based on:
    1. If destination doesn't exist
    2. If destination file size is less than min_size bytes
    3. If force flag is set

    Args:
        src_path: Source file path
        dest_path: Destination file path
        min_size: Minimum expected file size in bytes
        force: Force conversion regardless of conditions

    Returns:
        bool: True if conversion is needed, False otherwise
    """
    if force:
        return True

    if not dest_path.exists():
        return True

    dest_size = dest_path.stat().st_size
    return dest_size < min_size

def conv_one_file(args):
    root, filepath, f, cf, ff, dither, bgr_mode, out_path, force, min_size = args
    try:
        name = filepath.stem
        dest_path = get_dest_path(root, filepath, out_path, name, ff)

        # Check if conversion is needed
        if not should_convert(filepath, dest_path, min_size, force):
            return filepath, "SKIPPED (size OK)", time.time()

        # Ensure output directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Perform conversion
        conv = Converter(
            filepath.as_posix(), name, dither, name2const[f], cf_palette_bgr_en=bgr_mode
        )

        c_arr = ""
        if f in ["true_color", "true_color_alpha", "true_color_chroma"]:
            conv.convert(name2const[cf], 1 if f == "true_color_alpha" else 0)
            c_arr = conv.format_to_c_array()
        else:
            conv.convert(name2const[f])

        file_conf = {
            "C": {"suffix": ".c", "mode": "w"},
            "BIN": {"suffix": ".bin", "mode": "wb"},
        }

        with open(dest_path, file_conf[ff]["mode"]) as fi:
            res = (
                conv.get_c_code_file(name2const[f], c_arr)
                if ff == "C"
                else conv.get_bin_file(name2const[f])
            )
            fi.write(res)

        # Verify file was created and has content
        if not dest_path.exists():
            return filepath, "ERROR: Failed to create output file", time.time()

        if dest_path.stat().st_size < min_size:
            dest_path.unlink()  # Remove file smaller than minimum size
            return filepath, f"ERROR: Generated file smaller than {min_size} bytes", time.time()

        return filepath, "SUCCESS", time.time()

    except Exception as e:
        # Clean up partial/failed output
        if 'dest_path' in locals() and dest_path.exists():
            dest_path.unlink()
        return filepath, f"ERROR: {str(e)}", time.time()

def process_result(result, start_time, file_count, failed_pics):
    filepath, status, end_time = result
    duration = (end_time - start_time) * 1000
    if status == "SUCCESS":
        print(f"{file_count:<5} {filepath} FINISHED {duration:.2f} ms")
        return 1
    elif status.startswith("SKIPPED"):
        print(f"{file_count:<5} {filepath} {status}")
        return 0
    else:
        print(f"{file_count:<5} {filepath} {status} {duration:.2f} ms")
        failed_pics.append(filepath)
        return 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filepath",
        type=str,
        nargs="+",
        help="images dir paths (or file paths) you wanna convert",
    )
    parser.add_argument(
        "-f",
        "-format",
        type=str,
        default="true_color",
        choices=[
            "true_color",
            "true_color_alpha",
            "true_color_chroma",
            "indexed_1",
            "indexed_2",
            "indexed_4",
            "indexed_8",
            "alpha_1",
            "alpha_2",
            "alpha_4",
            "alpha_8",
            "raw",
            "raw_alpha",
            "raw_chroma",
        ],
        help="converted file format",
    )
    parser.add_argument(
        "-cf",
        "-color-format",
        type=str,
        default="RGB888",
        choices=["RGB332", "RGB565", "RGB565SWAP", "RGB888"],
        help="converted color format",
    )
    parser.add_argument(
        "-ff",
        "-file-format",
        type=str,
        default="C",
        choices=["C", "BIN"],
        help="converted file format: C(*.c), BIN(*.bin)",
    )
    parser.add_argument(
        "-o",
        "-output-filepath",
        type=str,
        default="",
        help="output file path",
    )
    parser.add_argument(
        "-r", action="store_const", const=True, help="convert files recursively"
    )
    parser.add_argument("-d", action="store_const", const=True, help="need to dith")
    parser.add_argument(
        "-b", action="store_const", const=True, default=True, help="BGR mode"
    )
    parser.add_argument(
        "--force", action="store_true", help="force conversion even if files are up to date"
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=196612,
        help="minimum size in bytes for output files (default: 196612)",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=mp.cpu_count(),
        help="number of parallel jobs (default: number of CPU cores)",
    )
    return parser.parse_args()

class Main(object):
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.output_path = Path(args.o)
        self.output_path.mkdir(exist_ok=True)
        self.file_count = 0
        self.failed_pic_paths = []
        self.start_time = time.time()

    def convert(self):
        print(f"using {self.args.jobs} processes")

        # Collect all files to process
        files_to_process = []
        for path in self.args.filepath:
            path = Path(path)
            if path.is_dir():
                path_glob = path.rglob if self.args.r else path.glob
                for file in path_glob("*.*"):
                    if check_allowed(file):
                        files_to_process.append((
                            path,
                            file,
                            self.args.f,
                            self.args.cf,
                            self.args.ff,
                            self.args.d,
                            self.args.b,
                            self.output_path,
                            self.args.force,
                            self.args.min_size
                        ))
            elif path.is_file() and check_allowed(path):
                files_to_process.append((
                    path.parent,
                    path,
                    self.args.f,
                    self.args.cf,
                    self.args.ff,
                    self.args.d,
                    self.args.b,
                    self.output_path,
                    self.args.force,
                    self.args.min_size
                ))

        # Process files in parallel
        with mp.Pool(processes=self.args.jobs) as pool:
            for result in pool.imap_unordered(conv_one_file, files_to_process):
                self.file_count += process_result(
                    result,
                    self.start_time,
                    self.file_count,
                    self.failed_pic_paths
                )

        print(f"\nConvert Complete. Total converted {self.file_count} file(s) in {time.time() - self.start_time:.2f} seconds.")

        if self.failed_pic_paths:
            print("\nFailed File List:")
            print(*self.failed_pic_paths, sep="\n")

if __name__ == "__main__":
    main = Main(parse_args())
    main.convert()
