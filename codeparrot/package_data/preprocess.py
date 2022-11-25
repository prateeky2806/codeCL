import os, sys

sys.path.insert(1, os.getcwd())
os.environ["HF_HOME"] = os.path.join("/nas-hdd/prateek/", ".cache/huggingface/")

import json, re
import jsonlines
import argparse
import numpy as np
import datasets
from collections import defaultdict as ddict


def process_categories(args):
    category = ["networking", "gui", "science", "web", "database", "other"]
    for filename in category:
        counts, skipped_lines = split_train_val_test(args, filename)
        print(
            f"Splitting file {filename}:  Train: {counts[0]} Val: {counts[1]} Test: {counts[2]} Skipped: {skipped_lines}"
        )
    return None


def split_train_val_test(args, filename):
    np.random.seed(args.seed)
    train_frac = 1 - args.val_frac - args.test_frac
    probs = [train_frac, args.val_frac, args.test_frac]
    line_dict = ddict(int)
    file_dict = {
        0: open(
            os.path.join(args.dump_dir, f"{filename.split('.')[0]}_train.jsonl"), "w"
        ),
        1: open(
            os.path.join(args.dump_dir, f"{filename.split('.')[0]}_val.jsonl"), "w"
        ),
        2: open(
            os.path.join(args.dump_dir, f"{filename.split('.')[0]}_test.jsonl"), "w"
        ),
    }
    with open(os.path.join(args.dump_dir, f"{filename}.jsonl"), "r") as read_file:
        skipped_lines = 0
        for line in read_file:
            file_number = np.random.multinomial(1, probs).argmax()
            if line_dict[file_number] > args.max_eval_size - 1:
                file_number = 0
            line = json.loads(line)
            content = line["content"]
            # File always starts with import statements.
            # Everything before import, from statements is ignored.
            line["content"] = content[
                min(content.find("import"), content.find("from")) :
            ]

            json.dump(line, file_dict[file_number])
            file_dict[file_number].write("\n")
            line_dict[file_number] += 1
    for f in file_dict.values():
        f.close()
    return line_dict, skipped_lines


def split_train_val_test_with_processing(filename, val_frac, test_frac, seed):
    np.random.seed(seed)
    train_frac = 1 - val_frac - test_frac
    probs = [train_frac, val_frac, test_frac]
    line_dict = ddict(int)
    file_dict = {
        0: open(f"{filename.split('.')[0]}_train.jsonl", "w"),
        1: open(f"{filename.split('.')[0]}_val.jsonl", "w"),
        2: open(f"{filename.split('.')[0]}_test.jsonl", "w"),
    }
    with open(f"{filename}.jsonl", "r") as read_file:
        skipped_lines = 0
        for line in read_file:
            file_number = np.random.multinomial(1, probs).argmax()
            line = json.loads(line)
            content = line["content"]
            # File always starts with import statements.
            # Everything before import, from statements is ignored.
            content = content[min(content.find("import"), content.find("from")) :]
            # Replace , followed by whitespaces to, single white space.
            content = re.sub(r",\s{2,}", ",", content)
            # Replace ( followed by whatespaces to (.
            content = re.sub(r"\(\s{2,}", "(", content)
            content = re.sub(r"\{\s{2,}", "{", content)
            content = re.sub(r"\[\s{2,}", "[", content)
            # Replace whitespace ) to ).
            content = re.sub(r"\s{2,}\)", ")", content)
            content = re.sub(r"\s{2,}\}", "}", content)
            content = re.sub(r"\s{2,}\]", "]", content)

            # Removes docstring.
            content = re.sub(r"'''[\d\D]*?'''", "", content)
            content = re.sub(r'"""[\d\D]*?"""', "", content)

            content = os.linesep.join([s for s in content.splitlines() if s])

            # Removes # comments and empty lines.
            final_str = ""
            for l in content.split("\n"):
                if l == "":
                    continue
                l = l.rstrip(" ")
                comment_idx = l.find("#")
                if comment_idx == -1:
                    final_str = final_str + l + "\n"
                else:
                    if l[:comment_idx].strip() is not "":
                        final_str = final_str + l[:comment_idx].rstrip(" ") + "\n"

            # Replace 4 spaces with tabs for indentation.
            final_str = final_str.replace(" " * 4, "\t")

            # File needs to have atleast three non empty lines.
            if final_str == "" or len(final_str.split("\n")) <= 2:
                skipped_lines += 1
                continue

            # If file is allocated to test, then split it at random location for creating line completion.
            if file_number == 2:
                split_str = final_str.split("\n")
                break_idx = np.random.randint(2, len(split_str))
                final_str = "\n".join(s for s in split_str[:break_idx])

            line = {"code": final_str}
            json.dump(line, file_dict[file_number])
            file_dict[file_number].write("\n")
            line_dict[file_number] += 1
    for f in file_dict.values():
        f.close()
    return line_dict, skipped_lines


def get_categories(args):
    science = [
        "scipy",
        "torch",
        "sklearn",
        "tensorflow",
        "pandas",
        "numpy",
        "matplotlib",
        "keras",
        "theano",
        "bokeh",
        "astropy",
        "seaborn",
        "networkx",
        "nltk",
        "skimage",
        "statsmodels",
        "transformers",
        "mxnet",
        "cv2",
        "scrapy",
        "math",
        "random",
        "astropy",
        "pyplot",
        "statistics",
        "pyarrow",
    ]
    web = [
        "django",
        "flask",
        "wsgiref",
        "html",
        "http",
        "lxml",
        "urlparse",
        "app",
        "starlette",
        "api",
        "urllib3",
        "aiohttp",
    ]
    database = [
        "sqlalchemy",
        "pymysql",
        "pymongo",
        "sql",
        "mysql",
        "sqlite3",
        "redis",
        "sqlobject",
        "database",
        "elasticsearch",
        "tabulate",
    ]
    networking = ["socket", "multiprocessing", "subprocess", "threading"]
    gui = ["swgpy", "gui"]

    patterns = ["from[ ]*[a-z0-9|.]*[ ]*import", "import[ ]*[a-z0-9|.]*[ ]*as"]

    code2pkg = dict()
    packages = dict()

    n_doc = 0

    science_files = []
    web_files = []
    database_files = []
    networking_files = []
    gui_files = []

    output = {
        "science": {},
        "web": {},
        "database": {},
        "networking": {},
        "gui": {},
        "other": {},
        "science_ids": set(),
        "web_ids": set(),
        "database_ids": set(),
        "networking_ids": set(),
        "gui_ids": set(),
        "other_ids": set(),
    }

    if "codeparrot" in args.dataset_name[0]:
        ds_list = []
        for dn in args.dataset_name:
            print(f"Loading dataset: {dn}")
            ds_list.append(datasets.load_dataset(dn, streaming=True, split="train"))
    elif "stack" in args.dataset_name[0]:
        licences = [
            "MIT-0",
            "MIT",
            "MIT-feh",
            "Apache-2.0",
            "BSD-3-Clause",
            "BSD-3-Clause-Clear",
            "BSD-3-Clause-No-Nuclear-License-2014",
            "BSD-2-Clause",
            "CC0-1.0",
            "EPL-1.0",
            "MPL-2.0",
            "Unlicense",
            "ISC",
            "Artistic-2.0",
            "deprecated_LGPL-3.0+",
            "deprecated_LGPL-2.1+",
            "ECL-2.0",
            "SHL-0.51",
            "MPL-2.0-no-copyleft-exception",
        ]
        ds_list = []
        for dn in args.dataset_name:
            print(f"Loading dataset: {dn}")
            ds_list.append(
                datasets.load_dataset(
                    dn,
                    data_dir=f"data/{args.lang}",
                    streaming=True,
                    split="train",
                    use_auth_token=True,
                )
            )

    for di, ds in enumerate(ds_list):
        for file in ds:
            if n_doc % 10000 == 0:
                print(f"Dataset:{di}\tProcessed {n_doc} Files!")
            content = file["content"]
            pkg = set()
            for pt in patterns:
                for seg in re.finditer(pt, content):
                    start = seg.span()[0]
                    end = seg.span()[1]
                    statement = content[start:end].replace(".", " ")

                    for top in statement.split(" "):
                        if len(top) > 0 and top not in ["from", "import"]:
                            # print(content[start:end], '------>', top)
                            if top not in pkg:
                                if top in science:
                                    output["science"][n_doc] = file
                                    output["science_ids"].add(n_doc)
                                elif top in web:
                                    output["web"][n_doc] = file
                                    output["web_ids"].add(n_doc)
                                elif top in database:
                                    output["database"][n_doc] = file
                                    output["database_ids"].add(n_doc)
                                elif top in networking:
                                    output["networking"][n_doc] = file
                                    output["networking_ids"].add(n_doc)
                                elif top in gui:
                                    output["gui"][n_doc] = file
                                    output["gui_ids"].add(n_doc)
                                else:
                                    output["other"][n_doc] = file
                                    output["other_ids"].add(n_doc)
                                pkg.add(top)
                                packages[top] = packages.get(top, 0) + 1
                            break
            code2pkg[n_doc] = pkg
            n_doc += 1
            if n_doc > args.max_lines:
                break

    category = ["science", "web", "database", "networking", "gui", "other"]
    overlap = np.zeros([len(category), len(category)]).astype(int)
    for i, k in enumerate(category):
        print(k, len(set(output[k + "_ids"])))
        for j, k1 in enumerate(category):
            overlap[i, j] = len(set(output[k + "_ids"]) & set(output[k1 + "_ids"]))

    print(overlap)

    # figure out overlaping and remove them.
    for k in ["science_ids", "web_ids", "database_ids", "networking_ids", "gui_ids"]:
        for k1 in [
            "science_ids",
            "web_ids",
            "database_ids",
            "networking_ids",
            "gui_ids",
        ]:
            if k1 != k:
                output[k] = output[k] - output[k1]

    for k in category:
        n = 0
        writers = jsonlines.open(os.path.join(args.dump_dir, f"{k}.jsonl"), mode="w")
        for kk, dd in output[k].items():
            if kk in output[f"{k}_ids"]:
                n += 1
                writers.write(dd)
        print(k, n)

    writer_ids = jsonlines.open(os.path.join(args.dump_dir, f"ids.jsonl"), mode="w")
    for k in [
        "science_ids",
        "web_ids",
        "database_ids",
        "networking_ids",
        "gui_ids",
        "other_ids",
    ]:
        writer_ids.write({k: list(output[k])})


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", nargs="+", required=True, help="")
    parser.add_argument("--lang", type=str, default="python", help="")
    parser.add_argument("--val_frac", type=float, default=0.1, help="")
    parser.add_argument("--test_frac", type=float, default=0.1, help="")
    parser.add_argument("--max_eval_size", type=int, default=5000, help="")
    parser.add_argument("--seed", type=int, default=28, help="")
    parser.add_argument(
        "--dump_dir", type=str, default="/nas-hdd/prateek/data/test", help=""
    )
    parser.add_argument("--get_categories", action="store_true")
    parser.add_argument("--process_split", action="store_true")
    parser.add_argument("--max_lines", type=int, default=1e15, help="")
    args = parser.parse_args()

    os.makedirs(args.dump_dir, exist_ok=True)
    if args.get_categories:
        get_categories(args)

    if args.process_split:
        print("Processing and splitting the categories files!")
        process_categories(args)
    print("finished!")


# python preprocess.py --dataset_name codeparrot/codeparrot-train-v2-near-dedup codeparrot/codeparrot-valid-v2-near-dedup --get_categories --process_split --dump_dir /nas-hdd/prateek/data/package-codeparrot-v2-near-dedup
# python preprocess.py --dataset_name bigcode/the-stack-dedup --get_categories --process_split --dump_dir /nas-hdd/prateek/data/package-the-stack-dedup
