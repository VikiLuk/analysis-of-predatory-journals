import chardet

# Detect the encoding of the file
with open('output.txt', 'rb') as f:
    result = chardet.detect(f.read())
    encoding = result['encoding']

# Open the file using the detected encoding
with open('output.txt', 'r', encoding=encoding) as f:
    text = f.read()

# Print the text
print(text)
