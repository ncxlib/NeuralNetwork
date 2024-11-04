import subprocess
import os
import nbformat as nbf


def clean_init_file(file_path):
    """
    Removes lines that start with 'from neuralnetwork.' from the given file.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    cleaned_lines = [
        line for line in lines if not line.startswith("from neuralnetwork.Layers")
    ]

    with open(file_path, "w") as f:
        f.writelines(cleaned_lines)


def add_init_files():
    """
    Recursively adds __init__.py files to all folders and subfolders in neuralnetwork
    if they don't already exist, then cleans up imports in neuralnetwork/__init__.py.
    """
    remove_all_init_files()

    base_dir = os.path.abspath(os.path.dirname(__file__))
    neuralnetwork_dir = os.path.join(base_dir, "neuralnetwork")

    for root, dirs, files in os.walk(neuralnetwork_dir):
        init_file = os.path.join(root, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                f.write("# Automatically generated __init__.py\n")
            print(f"Added __init__.py to {root}")
        else:
            print(f"__init__.py already exists in {root}")

    subprocess.run(
        ["mkinit", ".", "--recursive", "--write"], cwd=neuralnetwork_dir, check=True
    )

    neuralnetwork_init = os.path.join(neuralnetwork_dir, "__init__.py")
    clean_init_file(neuralnetwork_init)
    print(f"Cleaned up {neuralnetwork_init}")


def fmt():
    """
    Runs the Black formatter on the entire project directory.
    """
    subprocess.run(["black", "."])


def remove_all_init_files():
    """
    Recursively removes all __init__.py files in the entire project directory.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    for root, dirs, files in os.walk(project_root):
        for file in files:
            if file == "__init__.py":
                init_file_path = os.path.join(root, file)
                os.remove(init_file_path)
                print(f"Removed {init_file_path}")


def notebook():
    """
    Creates a notebook for quick testing in the scrap-testing folder

    Parameters:
    - notebook_dir (str): Directory where the notebook will be created.
    - notebook_name (str): Name of the notebook file.
    """

    notebook_dir = "scrap-testing"
    notebook_name = "test_notebook.ipynb"
    os.makedirs(notebook_dir, exist_ok=True)

    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell("# Scrap Testing Notebook\n"),
        nbf.v4.new_code_cell("import numpy as np\n"),
    ]

    notebook_path = os.path.join(notebook_dir, notebook_name)

    with open(notebook_path, "w") as f:
        nbf.write(nb, f)

    print(f"Created notebook at {notebook_path}")
