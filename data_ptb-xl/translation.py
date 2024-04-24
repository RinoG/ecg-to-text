"""
Based on the work by scro3517 on GitHub https://github.com/danikiyasseh/RTLP/blob/main/translate_ptbxl_text.py.
Modifications were made to enhance functionality and tailor it to specific needs.
This version includes changes to translation handling and data processing as described below:
- Only translate to English, remove other translations
- Change hard_code_text_changes_df to the end
- Add 'ekg' to 'ecg' in hard_code_text_changes_df

Original Author: scro3517
Date Created: October 27, 2020
Modifications by: RinoG
Date Modified: April 15, 2024
"""
import os
import time

import numpy as np
from tqdm import tqdm
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

from googletrans import Translator
from langdetect import detect

translator = Translator()

# %%

def load_ptbxl_df(basepath):
    """ Load Database With Text """
    df = pd.read_csv(os.path.join(basepath, 'val.csv'), index_col='ecg_id')
    return df


def make_changes_to_string(text):
    if 'sinusrhythmus' in text:
        text = text.replace('sinusrhythmus', 'sinusrhythm')

    if 'normales' in text:
        text = text.replace('normales', 'normal')

    if 'ekg' in text:
        text = text.replace('ekg', 'ecg')

    return text


def hard_code_text_changes_df(df):
    df.report = df.report.apply(make_changes_to_string)
    return df


# %%
def obtain_translations_df(df, dest_lang='en'):
    """Translate text into the desired language, avoiding duplicate translations."""
    # Set up an empty DataFrame to populate with translations
    translations_df = pd.DataFrame(index=df.index, columns=['translation'])

    # Set to store already translated texts
    translated_texts = {}

    # Process each report
    idx = 0
    for report in tqdm(df['report']):
        if report.strip():  # Check if the report is not empty
            # Check if the report has already been translated
            if report not in translated_texts:
                # Translate the report
                try:
                    translation_result = translator.translate(report, src='auto', dest=dest_lang)
                    text = translation_result.text
                    translated_texts[report] = text  # Store the translation in the dictionary
                    time.sleep(1)
                except Exception as e:
                    print(f"{idx} failed to translate: {report}")
                    text = report
            else:
                text = translated_texts[report]  # Retrieve the previously translated text
        else:
            text = ''  # No text to translate

        # Populate the DataFrame with the translation
        translations_df.at[idx, 'translation'] = text
        idx += 1

    return translations_df


def expand_df(df, translations_df, dest_lang):
    """ Append Translations As New Column in DF """
    new_column_name = f'{dest_lang}_report'
    df[new_column_name] = translations_df
    return df
# %%
def translate(base_path, dest_lang='en'):
    # dest_lang_list = ['it','pt','es','el','ja','zh-CN','en','de','fr']

    # """ Round 1 """
    # df = load_ptbxl_df(base_path)
    # # """ Hard Code Some Changes to Original Report """
    # # df = hard_code_text_changes_df(df)
    # """ Identify Destination Languages """
    # translations_df = obtain_translations_df(df, dest_lang)
    # df = expand_df(df, translations_df, dest_lang)
    # """ Save The Translations DataFrame (Frequently To Avoid Losing Progress) """
    # translations_df.to_csv(os.path.join(base_path, f'{dest_lang}_df.csv'))
    # # """ Save The Multi-lingual DataFrame (Frequently To Avoid Losing Progress) """
    # # df.to_csv(os.path.join('/mnt/SecondaryHDD/PTB-XL','multi_lingual_df.csv'))

    """ Round 2 """

    """ Round 2 of Translations to Catch Those Not Translated The First Time Round """
    """ Load Translated Report """
    translations_df = pd.read_csv(base_path + f'{dest_lang}_df.csv', index_col='ecg_id')
    translations_df.columns = ['report']
    translations_df['report'] = translations_df['report'].astype(str)

    """ Identify Reports Where Detected Language != Desired Language """
    indices_to_translate = []
    r = 0
    for text in tqdm(translations_df.report):
        if isinstance(text, str):
            # cannot identify correct language accurately,
            # but can be used to identify whether sentence is entirely in desired language
            src_lang = detect(text)
            if src_lang != dest_lang.lower():
                indices_to_translate.append(r)
        r += 1

    """ Translate Only Those That Need Translating """
    for index in tqdm(indices_to_translate):
        text = translations_df.report.iloc[index]
        translation = translator.translate(text, src='de', dest=dest_lang).text
        translations_df.iloc[index] = translation

    """ Hard Code Some Changes to Original Report """
    translations_df = hard_code_text_changes_df(translations_df)

    """ Save Translations DF """
    translations_df.to_csv(os.path.join(base_path, f'{dest_lang}_df_round4.csv'))


if __name__ == '__main__':
    translate('')