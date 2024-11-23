import subprocess
import os
import nbformat as nbf
import toml

def add_init_files():
    """
    Recursively adds __init__.py files to all folders and subfolders in specified directories
    if they don't already exist, then cleans up imports in neuralnetwork/__init__.py.
    """
    remove_all_init_files()

    base_dir = os.path.abspath(os.path.dirname(__file__))
    directories = [
        "models",
        "models/neuralnetwork",
        "models/classifiers",
        "datasets",
        "dataloaders",
        "preprocessing",
        "generators",
        "util",
        "preloaded",
        "evaluation",
        "visualization",
        "activations",
        "losses",
        "optimizers",
        "initializers"
    ]
    directories = [f"ncxlib/{file}" for file in directories]



    for directory in directories:
        target_dir = os.path.join(base_dir, directory)

        for root, dirs, files in os.walk(target_dir):
            init_file = os.path.join(root, "__init__.py")
            if not os.path.exists(init_file):
                with open(init_file, "w") as f:
                    f.write("# Automatically generated __init__.py\n")
                print(f"Added __init__.py to {root}")
            else:
                print(f"__init__.py already exists in {root}")

        # Generate __init__.py files with mkinit and clean imports
        subprocess.run(
            [
                "mkinit",
                ".",
                "--recursive",
                "--write",
                "--nomods",
                "--relative",
                "--black",
            ],
            cwd=target_dir,
            check=True,
        )

        update_init_files(directory)

    move_layer_import_to_top("ncxlib/models/neuralnetwork/layers/__init__.py", "Layer")
    move_layer_import_to_top("ncxlib/losses/__init__.py", "LossFunction")
    move_layer_import_to_top("ncxlib/initializers/__init__.py", "Initializer")
    move_layer_import_to_top("ncxlib/models/neuralnetwork/dataloaders/__init__.py", "Dataloader")
    move_layer_import_to_top("ncxlib/models/__init__.py", "Model")


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
    Creates a notebook for quick testing in the scrap-testing folder.

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


def get_import_dependencies(module_name, dir):
    module_path = os.path.join(dir, f"{module_name}.py")
    dependencies = []

    if os.path.exists(module_path):
        with open(module_path, "r") as f:
            for line in f:
                match = "from" in line and "import" in line
                if match:
                    dependency = line.split(" ")[3]
                    many_dependancies = dependency.split(",")
                    if len(many_dependancies) == 1:
                        if many_dependancies[0].strip().lower() == "":
                            continue
                        dependencies.append(many_dependancies[0].strip().lower())
                    else:
                        for each in many_dependancies:
                            if each.strip() == "":
                                continue
                            dependencies.append(each.strip().lower())

    return dependencies


def get_import_order(file_path, dir):
    with open(file_path, "r") as f:
        lines = f.readlines()

    imports = {}
    current_import = []
    inside_import_block = False

    # print(f"Processing file: {file_path}")
    for line in lines:
        if line.strip().startswith("from .") and "import (" in line:
            current_import.append(line.strip())
            inside_import_block = True
            module_name = line.split()[1][1:]
            # print(f"Found module: {module_name} in line {lines.index(line) + 1}.")
            continue

        if inside_import_block:
            current_import.append(line.strip())
            if line.strip() == ")":
                full_import_line = "\n".join(current_import)
                module_path = os.path.join(dir, module_name)
                depth = len(module_path.split(os.sep))
                imports[module_name] = (depth, full_import_line)
                current_import = []
                inside_import_block = False

    ordered_imports = []
    visited = set()

    def dfs(module):
        if module in visited:
            return
        visited.add(module)
        dependencies = get_import_dependencies(module, dir)

        for dep in dependencies:
            if dep in imports:
                dfs(dep)

        ordered_imports.append(imports[module][1])

    for module in imports:
        if module not in visited:
            dfs(module)

    return ordered_imports


def update_init_files(dir):
    for root, dirs, files in os.walk(dir):
        if "__init__.py" in files:
            init_file = os.path.join(root, "__init__.py")
            ordered_lines = get_import_order(init_file, dir)

            with open(init_file, "w") as f:
                # f.write("# Automatically ordered imports\n")
                f.write("\n".join(ordered_lines) + "\n")


def move_layer_import_to_top(init_path, module_name):
    """
    Moves the .layer import to the top of neuralnetwork/layers/__init__.py file,
    respecting multi-line Black-formatted import structure.
    """

    if os.path.exists(init_path):
        with open(init_path, "r") as f:
            data = f.read()

        layer_import = f"from .{module_name.lower()} import ( {module_name}, )\n"
        data = data.replace(
            f"from .{module_name.lower()} import (\n{module_name},\n)", ""
        )

        with open(init_path, "w") as f:
            f.writelines(layer_import + data)

        print(f"Moved .{module_name.lower()} imports to the top in {init_path}")


def increment_pypi_version(version):
    major, minor, patch = map(int, version.split('.'))
    if patch < 9:
        patch += 1
    else:
        patch = 0
        if minor < 9:
            minor += 1
        else:
            minor = 0
            major += 1
    return f"{major}.{minor}.{patch}"


def update_version_in_pyproject():
    with open("pyproject.toml", "r") as f:
        pyproject = toml.load(f)

    current_version = pyproject["tool"]["poetry"]["version"]
    new_version = increment_pypi_version(current_version)
    pyproject["tool"]["poetry"]["version"] = new_version

    with open("pyproject.toml", "w") as file:
        toml.dump(pyproject, file)
    print(f"Version updated from {current_version} to {new_version}")

def remove_poetry_lock():
    if os.path.exists("poetry.lock"):
        os.remove("poetry.lock")
        print("poetry.lock file removed.")

def run_poetry_commands():
    remove_poetry_lock()
    update_version_in_pyproject()

    subprocess.run(["poetry", "install"], check=True)
    subprocess.run(["poetry", "run", "gen"], check=True)
    subprocess.run(["poetry", "publish", "--build"], check=True)