import re
import docx2txt


def extract_text(file_path):
    doc = docx2txt.process(file_path)
    text = ''

    # Find the index of the "introduction" section
    intro_start_idx = doc.lower().find('introduction')
    refs_start_idx = re.search(r'\n\s*REF(?:ERENCES|RENCES|ERENCE):?\s*\n', doc, flags=re.IGNORECASE)

    if intro_start_idx >= 0:
        intro_text = doc[intro_start_idx:]

        # Remove tables
        intro_text = re.sub(r'\+-+\+\n(?:\|.*?\|)+\n\+-+\+', '', intro_text)  # matches simple tables
        intro_text = re.sub(r'\+-+\+\n\|(?:\s*\S+\s*\|)+\n\+-+\+\n(?:\|(?:\s*\S+\s*\|)+\n)+\+-+\+', '',
                            intro_text)  # matches tables with headers
        intro_text = re.sub(r'\+-+\+\n\|(?:\s*\S+\s*\|)+\n\+(?:-+\+){2,}\n(?:\|(?:\s*\S+\s*\|)+\n)+\+-+\+', '',
                            intro_text)  # matches tables with both headers and footers

        # Split document on "REFERENCES" or any variation of it
        sections = re.split(r'\nREF(?:ERENCES|RENCES|ERENCE):?\n', intro_text, flags=re.IGNORECASE)

        if refs_start_idx:
            refs_text = doc[refs_start_idx.end():]

            # Remove page numbers
            refs_text = re.sub(r'\n\s*\d+\s*\n', '\n', refs_text)

            # Remove footer
            refs_text = re.sub(r'(\n|\s)*(Available online at: https://ijcnis.org)(\n|\s)*', '', refs_text)

            refs_text = re.sub(r'\|\s*\d+\s*\|', '', refs_text)  # remove table elements

            refs_text = re.sub(r'\+-+\+\n(?:\|.*?\|)+\n\+-+\+', '', refs_text)  # matches simple tables
            refs_text = re.sub(r'\+-+\+\n\|(?:\s*\S+\s*\|)+\n\+-+\+\n(?:\|(?:\s*\S+\s*\|)+\n)+\+-+\+', '',
                                refs_text)  # matches tables with headers
            refs_text = re.sub(r'\+-+\+\n\|(?:\s*\S+\s*\|)+\n\+(?:-+\+){2,}\n(?:\|(?:\s*\S+\s*\|)+\n)+\+-+\+', '',
                                refs_text)  # matches tables with both headers and footers

            # Remove unnecessary spaces and write to file
            refs_text = re.sub(r'\s+', ' ', refs_text).strip()
            with open('references.txt', 'w', encoding='utf-8') as f:
                f.write(refs_text)

        # Only keep text from the first section (the "introduction" section)
        if sections:
            intro_text = sections[0]
            intro_text = re.sub(r'\$.*?\$', '', intro_text)  # remove math expressions
            intro_text = re.sub(r'\|\s*\d+\s*\|', '', intro_text)  # remove table elements
            intro_text = re.sub(r'(\d+:)?\s*\S*\s*[\u22a2-\u22b3\u2190-\u2194\u2196-\u2199]\s*\S*\s*(\d+:)?', '',
                                intro_text)  # remove mathematical expressions
            intro_text = re.sub(r'\n\s*\n', '\n', intro_text)  # remove empty lines
            intro_text = re.sub(r'\s+', ' ', intro_text)  # remove unnecessary spaces
            text = intro_text.strip()

    return text


if __name__ == '__main__':
    file = 'IJCNIS-V12N1-142'
    file_path = 'docx/IJCNIS/' + file + '.docx'
    output_file_path = 'output.txt'
    text = extract_text(file_path)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.write(text)
