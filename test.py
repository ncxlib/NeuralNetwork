import os

base_dir = 'dataloaders'

def get_import_dependencies(module_name):
    module_path = os.path.join(base_dir, f"{module_name}.py")
    dependencies = []
    
    if os.path.exists(module_path):
        with open(module_path, 'r') as f:
            for line in f:
                match = "from" in line and "import" in line
                if match:
                    print(f"isMatch: {match} | Line: {line}")
                    dependency = line.split(" ")[3]
                    many_dependancies = dependency.split(",")
                    if len(many_dependancies) == 1:
                        if many_dependancies[0].strip().lower() == "": 
                            continue
                        dependencies.append(many_dependancies[0].strip().lower())
                    else:
                        for each in many_dependancies:
                            if each.strip() == "": continue
                            dependencies.append(each.strip().lower())
                    
    return dependencies

def get_import_order(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    imports = {}
    current_import = []
    inside_import_block = False

    print(f"Processing file: {file_path}")
    for line in lines:
        if line.strip().startswith("from .") and "import (" in line:
            current_import.append(line.strip())
            inside_import_block = True
            module_name = line.split()[1][1:]
            print(f"Found module: {module_name} in line {lines.index(line) + 1}.")
            continue

        if inside_import_block:
            current_import.append(line.strip())
            if line.strip() == ")":
                full_import_line = "\n".join(current_import)
                module_path = os.path.join(base_dir, module_name)
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
        dependencies = get_import_dependencies(module)
        print(f"Dependancies inside {module} are {dependencies}")
        
        for dep in dependencies:
            if dep in imports:
                dfs(dep)
        
        ordered_imports.append(imports[module][1])

    for module in imports:
        if module not in visited:
            dfs(module)

    return ordered_imports

def update_init_files():
    for root, dirs, files in os.walk(base_dir):
        if '__init__.py' in files:
            init_file = os.path.join(root, '__init__.py')
            ordered_lines = get_import_order(init_file)

            with open(init_file, 'w') as f:
                f.write("# Automatically ordered imports\n")
                f.write('\n'.join(ordered_lines) + '\n')

if __name__ == "__main__":
    update_init_files()
