#!/usr/bin/env python3
"""
Simple script to add "#" to comment lines in all event files in RawEventData.
"""

import os

def fix_comments():
    """Add '#' to comment lines in all .txt files in RawEventData."""
    
    folder = "Data/RawEventData/"
    
    # Get all .txt files
    files = [f for f in os.listdir(folder) if f.endswith('.txt')]
    
    updated_count = 0
    
    for file_name in files:
        file_path = folder + file_name
        updated_lines = []
        needs_update = False
        
        with open(file_path, 'r') as file:
            lines = file.readlines()
            
        for line in lines:
            stripped = line.strip()
            
            # Skip empty lines
            if not stripped:
                updated_lines.append(line)
                continue
            
            # If line starts with a number, it's data - keep as is
            parts = stripped.split(", ")
            if parts[0].isnumeric():
                updated_lines.append(line)
            else:
                # It's a comment line - add "#" if not already there
                if not stripped.startswith('#'):
                    updated_lines.append('# ' + line)
                    needs_update = True
                else:
                    updated_lines.append(line)
        
        # Write back if changes were made
        if needs_update:
            with open(file_path, 'w') as file:
                file.writelines(updated_lines)
            updated_count += 1
            print(f"Updated: {file_name}")
    
    print(f"\nUpdated {updated_count} files out of {len(files)} total files.")

if __name__ == "__main__":
    fix_comments() 