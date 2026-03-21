import sys
import re
import os

def extract_code(filepath):
    if not os.path.exists(filepath):
        print(f"File {filepath} not found")
        return ""
    with open(filepath, 'r') as f:
        content = f.read()
    matches = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
    return '\n'.join(matches)

def main():
    files = {
        'alggeom2.md': 'scripts/alggeom2_demo.py',
        'markov.md': 'scripts/markov_demo.py',
        'numa_alggeo.md': 'scripts/numa_alggeo_demo.py'
    }
    for md_file, py_file in files.items():
        code = extract_code(md_file)
        if code:
            with open(py_file, 'w') as f:
                f.write(code)
            print(f"Extracted code from {md_file} to {py_file}")

if __name__ == "__main__":
    main()
