import re

class OCRCleaner:
    def __init__(self):
        # Domain-aware deterministic corrections
        self.corrections = {
            r'\bboller\b': 'boiler',
            r'\bfumace\b': 'furnace',
            r'\befficlency\b': 'efficiency',
            r'\bIoss\b': 'loss',
            r'\b0xygen\b': 'oxygen',
            r'\bkwh\b': 'kWh',
            r'\bkw\b': 'kW',
            r'\bkg/cm2\b': 'kg/cm²',
            r'\bdeg C\b': '°C',
            r'\bdegC\b': '°C'
        }

    def clean_text(self, text: str) -> str:
        """Applies OCR cleanup rules to the raw scanned text."""
        if not text:
            return ""

        # Normalize whitespace
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Apply deterministic corrections
        for pattern, replacement in self.corrections.items():
            text = re.sub(pattern, replacement, text)

        # Remove strange isolated artifacts while preserving numbers
        text = "\n".join([line for line in text.split("\n") if not re.match(r'^[^a-zA-Z0-9]+$', line.strip())])

        return text.strip()

ocr_cleaner = OCRCleaner()
