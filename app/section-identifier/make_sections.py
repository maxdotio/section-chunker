import hashlib
import re

def fix_dollars(text):
    pattern = r'\$[\d\,\.\s]+'
    matches = re.findall(pattern, text)
    if matches:
        for match in matches:
            fixed_match = re.sub(r'\s+', '', match) + ' '
            text = text.replace(match, fixed_match)
    return text

# Clean up extra spaces and other artifacts.
def clean_source(text):
    text = fix_dollars(text)
    return (text
            .replace(r'(\s*\n)+', '\n')
            .replace(r'(,\s+)+', ', ')
            .replace(r'( )+', ' ')
            .replace(r'(\$\s*)+', '$')
            .strip())

def generate_hash(input_string):
    return hashlib.sha256(input_string.encode()).hexdigest()

def add_node(level, header, text, paragraphids, pages):
    id = generate_hash(text)
    return {"id": id, "level": level, "header": header, "text": text, "paragraphids": paragraphids, "pages": pages}

def get_sections(metadata, predictions):
    current_level = 0
    form = 0
    candidates = []
    lines = []
    for page in metadata["contract"]["pages"]:
        for line in page["lines"]:
            lines.append(line["content"])

    for i in range(len(lines)):
        l = lines[i]
        content = l.strip()
        header = ""
        isheader = False

        if predictions[i] == 1:
            header = content
            isheader = True


        content = clean_source(content)
        candidates.append({"level": current_level, "line": content, "form": form, "isheader": isheader, "header": header, "lineid": i})

    parsed = candidates
    tree = []
    row = parsed[0]
    accumulated = [parsed[0]['line']]
    lineids = [parsed[0]['lineid'] or 0]
    pages = [0]
    current_level = parsed[0]['level']
    current_header = parsed[0]['line']

    for i in range(1, len(parsed)):
        row = parsed[i]
        if row['isheader']:
            text = '\n'.join(accumulated)
            node = add_node(current_level, current_header, text, lineids, pages)
            tree.append(node)
            accumulated = []
            lineids = []
            pages = []
            current_level = row['level']
            current_header = row['header']
        
        accumulated.append(row['line'])
        lineids.append(row['lineid'])

    if row:
        text = '\n'.join(accumulated)
        node = add_node(current_level, current_header, text, lineids, pages)
        tree.append(node)

    sections = [node["text"] for node in tree]

    return sections
    