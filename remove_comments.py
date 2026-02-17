"""
Run this script from your project root to remove all comments.
Usage: python remove_comments.py
"""

import os
import re
import ast

def remove_comments_from_file(filepath):
    """Remove all # comments and standalone docstrings from Python file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    cleaned_lines = []
    
    in_multiline_string = False
    multiline_char = None
    
    for line in lines:
        stripped = line.strip()
        
        # Track multiline strings (keep them - they might be actual strings)
        triple_double = line.count('"""')
        triple_single = line.count("'''")
        
        if not in_multiline_string:
            if triple_double % 2 != 0:
                in_multiline_string = True
                multiline_char = '"""'
            elif triple_single % 2 != 0:
                in_multiline_string = True
                multiline_char = "'''"
        else:
            if multiline_char in line:
                in_multiline_string = False
                multiline_char = None
            cleaned_lines.append(line)
            continue
        
        # Skip pure comment lines
        if stripped.startswith('#'):
            continue
        
        # Remove inline comments (but not # inside strings)
        cleaned_line = remove_inline_comment(line)
        
        # Skip lines that became empty after removing comment
        if cleaned_line.strip() == '' and stripped.startswith('#'):
            continue
            
        cleaned_lines.append(cleaned_line)
    
    # Remove excessive blank lines (max 2 consecutive)
    final_lines = []
    blank_count = 0
    for line in cleaned_lines:
        if line.strip() == '':
            blank_count += 1
            if blank_count <= 2:
                final_lines.append(line)
        else:
            blank_count = 0
            final_lines.append(line)
    
    cleaned_content = '\n'.join(final_lines)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    print(f"✓ Cleaned: {filepath}")


def remove_inline_comment(line):
    """Remove inline # comments, respecting strings."""
    in_string = False
    string_char = None
    i = 0
    
    while i < len(line):
        char = line[i]
        
        if not in_string:
            if char in ('"', "'"):
                # Check for triple quote
                if line[i:i+3] in ('"""', "'''"):
                    in_string = True
                    string_char = line[i:i+3]
                    i += 3
                    continue
                else:
                    in_string = True
                    string_char = char
            elif char == '#':
                # Found comment - return everything before it (rstrip)
                return line[:i].rstrip()
        else:
            if isinstance(string_char, str) and len(string_char) == 3:
                if line[i:i+3] == string_char:
                    in_string = False
                    string_char = None
                    i += 3
                    continue
            else:
                if char == string_char and (i == 0 or line[i-1] != '\\'):
                    in_string = False
                    string_char = None
        i += 1
    
    return line


def process_directory(directory, extensions=('.py',)):
    """Process all Python files in directory."""
    count = 0
    for root, dirs, files in os.walk(directory):
        # Skip hidden dirs, venv, cache
        dirs[:] = [d for d in dirs if not d.startswith('.') 
                   and d not in ('venv', 'venv310', '__pycache__', 
                                  'node_modules', '.git', 'env')]
        
        for file in files:
            if file.endswith(extensions):
                filepath = os.path.join(root, file)
                try:
                    remove_comments_from_file(filepath)
                    count += 1
                except Exception as e:
                    print(f"✗ Error processing {filepath}: {e}")
    
    return count


if __name__ == "__main__":
    print("=" * 50)
    print("🧹 REMOVING COMMENTS FROM ALL PYTHON FILES")
    print("=" * 50)
    
    # Process these folders only
    folders_to_clean = [
        'app',
        'feature_pipeline', 
        'training_pipeline'
    ]
    
    total = 0
    for folder in folders_to_clean:
        if os.path.exists(folder):
            print(f"\nProcessing {folder}/...")
            count = process_directory(folder)
            total += count
        else:
            print(f"⚠ Folder not found: {folder}")
    
    print(f"\n✅ Done! Cleaned {total} files.")
    print("\nNext steps:")
    print("  git add .")
    print('  git commit -m "Clean: Remove comments from codebase"')
    print("  git push")