import re

INPUT_FILE = "data/corpus.txt"
OUTPUT_FILE = "data/corpus_clean.txt"

def main():
    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        content = f.read()

    # Lowercase everything
    print("Converting to lowercase...")
    content = content.lower()

    # Remove carriage returns
    print("Removing carriage returns...")
    content = content.replace("\r", "")

    # Split into books based on START marker
    # The first chunk might be junk before the first book, but often it's the first book's header.
    # We'll split and discard the "preamble" if it doesn't look like a book start,
    # but the START marker itself is a good delimiter.
    # Note: '*** start of' will be lowercase now.
    
    print("Splitting into books...")
    parts = re.split(r'\*\*\* start of .*?\*\*\*', content)
    
    # The first part is usually the header/preamble of the first book BEFORE the first START marker 
    # OR just empty if the file starts with the marker. 
    # Actually, Gutenberg files usually have a relatively small preamble before the FIRST start marker.
    # But subsequent books in a combined file might be concatenated directly.
    # Let's see... if I split by START, the text of the book follows the split.
    # part[0] is preamble before 1st book. part[1] is 1st book, etc.
    
    cleaned_parts = []
    
    # We skip part[0] because it's usually the preamble of the very first book or just empty space
    # However, we should check if there is valid content there. 
    # Given the previous `head` output, the file STARTs with "The Project Gutenberg eBook...", 
    # and the START marker appears at line 28.
    # So part[0] is definitely the first header. We should probably discard it as "boilerplate".
    
    for i, part in enumerate(parts):
        if i == 0:
            continue # Skip the first chunk (pre-start header)
            
        # Now clean valid book parts
        
        # 1. Trimming the End
        # Look for END marker
        end_marker_match = re.search(r'\*\*\* end of .*?\*\*\*', part)
        if end_marker_match:
            # Cut off everything after and including the marker
            part = part[:end_marker_match.start()]
        else:
            # If no explicit END marker, look for valid footers or next headers
            
            # "End of the Project Gutenberg"
            footer_match = re.search(r'end of the project gutenberg', part)
            if footer_match:
                part = part[:footer_match.start()]
            else:
                 # Fallback: look for license block start if it appears at the end
                 license_match = re.search(r'start: full license', part)
                 if license_match:
                     part = part[:license_match.start()]
                 else:
                     # Check for the Header of the NEXT book which might be stuck at the end of this chunk
                     # "The Project Gutenberg eBook"
                     header_match = re.search(r'the project gutenberg ebook', part)
                     if header_match:
                         part = part[:header_match.start()]
                     
                     # Also "End of Project Gutenberg's"
                     end_pg_match = re.search(r"end of project gutenberg's", part)
                     if end_pg_match:
                         part = part[:end_pg_match.start()]

        # 2. Line-by-line processing
        lines = part.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove visual separators junk lines (e.g. lines with repeated =, *, or -)
            # Regex: line consisting only of whitespace and one of these chars repeated (allowing spaces)
            # Example: " * * * " or "======"
            if re.match(r'^\s*([=\*\-]\s*)+$', line):
                continue
            
            # Remove "Produced by..." or "Transcribed by..." or "Credits:" lines near the start/end
            # These are common in the first few lines after the start marker
            if re.match(r'^\s*(produced|transcribed) by', line):
                continue
            if re.match(r'^\s*credits:', line):
                continue
            
            cleaned_lines.append(line)
        
        part = '\n'.join(cleaned_lines)
        cleaned_parts.append(part)

    print("Joining parts...")
    full_text = "\n\n".join(cleaned_parts)

    # Collapse 3 or more consecutive newlines into 2
    print("Collapsing newlines...")
    full_text = re.sub(r'\n{3,}', '\n\n', full_text)
    
    print(f"Writing to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(full_text)
        
    print("Done.")

if __name__ == "__main__":
    main()
