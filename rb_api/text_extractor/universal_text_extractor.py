import textract

def extract_raw_text(path_to_file: str) -> str:
    raw_text = textract.process(path_to_file).decode('utf-8')
    return raw_text
