import pandas as pd
import os
import re
import nltk

from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)
base_stopwords = set(stopwords.words('english'))

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

GREEK_TO_ENGLISH = {
    'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta', 'ε': 'epsilon',
    'ζ': 'zeta', 'η': 'eta', 'θ': 'theta', 'ι': 'iota', 'κ': 'kappa',
    'λ': 'lambda', 'μ': 'micro',  
    'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron', 'π': 'pi', 'ρ': 'rho',
    'σ': 'sigma', 'ς': 'sigma', 'τ': 'tau', 'υ': 'upsilon', 'φ': 'phi',
    'χ': 'chi', 'ψ': 'psi', 'ω': 'omega',
}

def clean_text(text, custom_stopwords):
    if pd.isna(text) or text is None:
        return ""
    
    text = str(text).lower()

    for greek_char, english_word in GREEK_TO_ENGLISH.items():
        text = text.replace(greek_char, english_word)
    
    text = re.sub(r'[^a-z0-9\s-]', ' ', text)
    
    text = re.sub(r'\s+', ' ', text).strip()
    
    words = text.split()
    custom_stopwords.update(["doi", "et", "al", "al", "figure", "table", "data", "used", "show", "found", "based", "study", "result"])
    cleaned_words = [w for w in words if w not in custom_stopwords]
    
    return " ".join(cleaned_words)

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    file_name = input("input document name (eg. papers.csv): ").strip()
    
    file_path = os.path.join(RAW_DIR, file_name)
    if not os.path.exists(file_path):
        print(f"Error: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {file_name}, {len(df)} lines")
    except Exception as e:
        print(f"{e}")
        return

    cols_input = input("Input column name to clean (separate by ,) :")
    cols_to_clean = [c.strip() for c in cols_input.split(',')]

    keep_input = input("Input columns to keep in output (separate by ,) or leave blank to keep all :")

    for col in cols_to_clean:
        if col in df.columns:
            new_col_name = f"cleaned_{col}"
            df[new_col_name] = df[col].apply(lambda x: clean_text(x, base_stopwords))
        else:
            print(f"'{col}' doesnt exist")

    if keep_input.strip():
        cols_to_keep = [c.strip() for c in keep_input.split(',')]
        
        for col in cols_to_clean:
            new_col = f"cleaned_{col}"
            if new_col in df.columns and new_col not in cols_to_keep:
                cols_to_keep.append(new_col)
                
        cols_to_keep = [c for c in cols_to_keep if c in df.columns]
        
        df = df[cols_to_keep]

    output_path = os.path.join(PROCESSED_DIR, f"cleaned_{file_name}")
    df.to_csv(output_path, index=False)
    print(f"Saved at {output_path}")

if __name__ == "__main__":
    main()