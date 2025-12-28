
import os
import re

def process_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    lines = content.split('\n')
    new_lines = []
    
    # Patterns for traits we want to hide completely
    noise_patterns = [
        r'IntoEither', r'SupersetOf', r'StructuralPartialEq', r'Scalar',
        r'TryFrom', r'TryInto', r'Borrow', r'ToOwned', r'Any', r'From', r'Into',
    ]

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # --- ARGUMENTS TABLE LOGIC ---
        # Matches "# Arguments" with any indentation
        if stripped == "# Arguments" or stripped == "## Arguments":
            # Consume the header (don't print it yet, we'll make a table header)
            # Actually, let's keep the header but make table below? 
            # User wants "those suppose be table".
            # If we make a table, we don't strictly need "# Arguments" if it's obvious, 
            # but usually we keep it.
            new_lines.append(line)
            new_lines.append("")
            
            args_data = []
            i += 1
            while i < len(lines):
                lx = lines[i]
                # Check for list item: * `track_thresh` - description
                m_arg = re.match(r'^\s*\*\s*`(.+?)`\s*-\s*(.+)', lx)
                if m_arg:
                    args_data.append([m_arg.group(1), m_arg.group(2)])
                elif lx.strip() == "":
                    # Empty line allowed
                    pass
                else:
                    # Non-empty, non-match line -> End of arguments section
                    # But wait, descriptions might wrap? 
                    # For now assume one-liners or simple structure as per view_file
                    
                    # If it looks like another section or code block, break
                    # Also break if it looks like a new list item that isn't an argument (e.g. method def)
                    if lx.lstrip().startswith('#') or lx.strip().startswith('```') or lx.strip().startswith('Returns'):
                         i -= 1
                         break
                    
                    if lx.lstrip().startswith('- <span'):
                         i -= 1
                         break

                    # If it's just text, maybe append to description?
                    if args_data:
                        args_data[-1][1] += " " + lx.strip()
                i += 1
            
            # Render Args Table
            if args_data:
                new_lines.append("| Argument | Description |")
                new_lines.append("|---|---|")
                for name, desc in args_data:
                    new_lines.append(f"| `{name}` | {desc} |")
                new_lines.append("")
            
            i += 1
            continue
            
        # --- FIELDS TABLE LOGIC ---
        if stripped == "#### Fields":
            new_lines.append(line)
            new_lines.append("")
            
            fields_data = [] # {name, type, desc}
            i += 1
            while i < len(lines):
                lx = lines[i]
                # Match: - **`name`**: `type` (allow indentation)
                m_field = re.match(r'^\s*-\s*\*\*`(.+?)`\*\*:\s*`(.+?)`', lx)
                
                if m_field:
                    fields_data.append({
                        'name': m_field.group(1),
                        'type': m_field.group(2),
                        'desc': []
                    })
                elif lx.strip() == "":
                    pass
                elif lx.lstrip().startswith('#'): 
                    # New header -> end of fields
                    i -= 1
                    break
                else:
                    # Description line
                    if fields_data:
                         fields_data[-1]['desc'].append(lx.strip())
                
                i += 1
            
            # Render Fields Table
            if fields_data:
                new_lines.append("| Name | Type | Description |")
                new_lines.append("|---|---|---|")
                for fd in fields_data:
                    d = " ".join(fd['desc']).replace("|", "\\|")
                    new_lines.append(f"| `{fd['name']}` | `{fd['type']}` | {d} |")
                new_lines.append("")
            
            i += 1
            continue

        # --- FIX LINKS ---
        if '| mod |' in line:
             line = re.sub(r'\[`(\w+)`\]\(#\1\)', r'[`\1`](\1/index.md#\1)', line)

        # --- TRAIT HEADER FORMATTING ---
        # Catch "##### `impl ...`" and convert to code block
        # Matches: ##### `impl Clone for STrack`
        m_head_impl = re.match(r'^#####\s*`impl\s+(.+)`', line)
        if m_head_impl:
             # Check for noise in the header itself
             text = m_head_impl.group(1)
             is_noise = any(re.search(p, text) for p in noise_patterns)
             if is_noise:
                 i += 1
                 continue
                 
             # Output as Rust code block
             new_lines.append(f"```rust")
             new_lines.append(f"impl {text}")
             new_lines.append(f"```")
             i += 1
             continue

        # Check for Noise Headers to skip section
        if line.lstrip().startswith('#'):
            is_noise = any(re.search(p, line) for p in noise_patterns)
            if is_noise:
                # Skip this line
                i += 1
                continue

        # --- METHOD CODE BLOCKS ---
        # - <span ...></span>`fn ...`
        m_fn = re.match(r'^\s*-\s*(<span id=".*?"></span>)(`fn .*?`)', line)
        if m_fn:
            span = m_fn.group(1)
            code = m_fn.group(2).strip('`')
            new_lines.append(span)
            new_lines.append("```rust")
            new_lines.append(code)
            new_lines.append("```")
            i += 1
            continue

        # --- TRAIT IMPL LIST ITEMS ---
        # - <span ...></span>`impl ...`
        m_impl = re.match(r'^\s*-\s*(<span id=".*?"></span>)(`impl .*?`)', line)
        if m_impl:
             span = m_impl.group(1)
             text = m_impl.group(2).strip('`')
             is_noise = any(re.search(p, text) for p in noise_patterns)
             if is_noise:
                 i += 1
                 continue
             
             new_lines.append(span)
             new_lines.append("```rust")
             new_lines.append(text)
             new_lines.append("```")
             i += 1
             continue

        new_lines.append(line)
        i += 1

    with open(filepath, 'w') as f:
        f.write('\n'.join(new_lines))

def walk_and_fix(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                process_file(os.path.join(root, file))

if __name__ == "__main__":
    walk_and_fix("docs/api")
    print("Fixed formatting in docs/api")
