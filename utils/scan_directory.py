import os
import ast

def extract_functions_with_docstrings(root_dir: str) -> dict:
    """
    Recursively scans the given root directory for Python files and extracts
    all function names along with their docstrings.

    Parameters:
    - root_dir (str): Path to the project directory.

    Returns:
    - dict: Mapping of file paths to lists of (function name, docstring) tuples.
    """
    results = {}

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".py"):
                file_path = os.path.join(dirpath, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    try:
                        tree = ast.parse(f.read(), filename=file_path)
                        functions = []
                        for node in ast.walk(tree):
                            if isinstance(node, ast.FunctionDef):
                                docstring = ast.get_docstring(node)
                                functions.append((node.name, docstring))
                        if functions:
                            results[file_path] = functions
                    except SyntaxError:
                        print(f"⚠️ Syntax error in file: {file_path}")

    return results

#import streamlit as st
print(extract_functions_with_docstrings(os.getcwd()))
#st.json(extract_functions_with_docstrings(os.getcwd()))