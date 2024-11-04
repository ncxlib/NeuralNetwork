import subprocess


def fmt():
    subprocess.run(["black", "."])
