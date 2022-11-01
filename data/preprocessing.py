import pandas as pd
import numpy as np


def clean_data(df):
    # Drop text features
    df.drop(columns=['description', 'name'], inplace=True)
    # prepare srm feature
    srm_over_40_idx = (pd.to_numeric(df[df.srm != 'Over 40'].srm) > 40).values
    num_srm = pd.to_numeric(df[df.srm != 'Over 40'].srm)
    srm_over_40_mean = int(num_srm[num_srm > 40].mean())
    srm_over_40_mean
    df.replace({'srm' : {'Over 40' : str(srm_over_40_mean)}}, inplace=True)
    df = df.astype({'srm' : 'int32'})
    # replace nan values of feature glass
    df.glass = df.glass.fillna('Nglass')
    # convert isOrganic column into numeric
    df.isOrganic = (df.isOrganic == 'Y').astype('int32')
    # clip originalGravity
    df.originalGravity = df.originalGravity.clip(upper=1.13)
    # limit ibu if train preparation
    if 'ibu' in df.columns:
        df.ibu = df.ibu.clip(upper=120)
    return df


def add_new_features(df):
    df['abv_mul_grav'] = df.abv * df.originalGravity
    df['abv_mul_srm'] = df.abv * df.srm
    df['srm_div_abv'] = df.srm / df.abv
    df['srm_mull_grav'] = df.srm * df.originalGravity
    df['srm_mull_grav_div_abv'] = df.srm * df.originalGravity / df.abv
    return df