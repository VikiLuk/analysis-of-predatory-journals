import re
import docx2txt


def extract_text(file_path):
    doc = docx2txt.process(file_path)
    text = ''

    # Find the index of the "introduction" section
    intro_start_idx = doc.lower().find('introduction')
    refs_start_idx = re.search(r'\n\s*REF(?:ERENCES|RENCES):?\s*\n', doc, flags=re.IGNORECASE)

    if intro_start_idx >= 0:
        intro_text = doc[intro_start_idx:]

        # Split document on "REFERENCES" or any variation of it
        sections = re.split(r'\nREF(?:ERENCES|RENCES):?\n', intro_text, flags=re.IGNORECASE)

        if refs_start_idx:
            refs_text = doc[refs_start_idx.end():]

            # Remove unnecessary spaces and write to file
            refs_text = re.sub(r'\s+', ' ', refs_text).strip()
            with open('references.txt', 'w', encoding='utf-8') as f:
                f.write(refs_text)

        # Only keep text from the first section (the "introduction" section)
        if sections:
            intro_text = sections[0]
            intro_text = re.sub(r'\$.*?\$', '', intro_text)  # remove math expressions
            intro_text = re.sub(r'\n\s*\n', '\n', intro_text)  # remove empty lines
            intro_text = re.sub(r'\s+', ' ', intro_text)  # remove unnecessary spaces
            text = intro_text.strip()

    return text


if __name__ == '__main__':
    file_path = 'docx/PatternRecognition/PR-Vol132-Dec22-108932.docx'
    output_file_path = 'output.txt'
    text = extract_text(file_path)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(text)
