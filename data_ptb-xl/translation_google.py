import os
import time

import numpy as np
from tqdm import tqdm
import pandas as pd
from google.cloud import translate_v2 as translate

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'google_cloud_translate_api_key.json'


def load_ptbxl_df(basepath):
    """ Load Database With Text """
    df = pd.read_csv(os.path.join(basepath, 'ptbxl_database.csv'), index_col='ecg_id')
    return df


def translate_text(target: str, text: str) -> str:
    """Translates text into the target language.

    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    translate_client = translate.Client()

    if isinstance(text, bytes):
        text = text.decode("utf-8")

    result = translate_client.translate(text, target_language=target)

    return result["translatedText"]


def obtain_translations_df(df, dest_lang='en'):
    translations_df = pd.DataFrame(index=df.index, columns=['translation'])

    translated_texts = {}
    for idx, data in tqdm(df.iterrows(), total=df.shape[0]):
        report = data.report
        if report.strip():
            if report not in translated_texts:
                try:
                    text = translate_text(dest_lang, report.strip())
                    translated_texts[report] = text
                    time.sleep(0.1)  # Sleep for 0.1 seconds between requests
                except Exception as e:
                    print(e)
                    print(f"{idx} failed to translate: {report}")
                    text = report
            else:
                text = translated_texts[report]
        else:
            text = ''
        translations_df.at[idx, 'translation'] = text
    return translations_df


def translate_dataset(base_path, dest_lang='en'):
    df = load_ptbxl_df(base_path)

    translations_df = obtain_translations_df(df, dest_lang)

    translations_df.to_csv(os.path.join(base_path, f'{dest_lang}_df.csv'))


if __name__ == '__main__':
    translate_dataset('')
