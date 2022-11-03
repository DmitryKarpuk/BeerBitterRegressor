SRM_OVER_MEAN = 60  # constant calculated in eda


def clean_data(df):
    # Drop text features
    df.drop(columns=["description", "name"], inplace=True)
    # prepare srm feature
    df.replace({"srm": {"Over 40": str(SRM_OVER_MEAN)}}, inplace=True)
    df = df.astype({"srm": "int32"})
    # replace nan values of feature glass
    df.glass = df.glass.fillna("Nglass")
    # convert isOrganic column into numeric
    df.isOrganic = (df.isOrganic == "Y").astype("int32")
    # clip originalGravity
    df.originalGravity = df.originalGravity.clip(upper=1.13)
    # limit ibu if train preparation
    if "ibu" in df.columns:
        df.ibu = df.ibu.clip(upper=120)
    return df


def add_new_features(df):
    df["abv_mul_grav"] = df.abv * df.originalGravity
    df["abv_mul_srm"] = df.abv * df.srm
    df["srm_div_abv"] = df.srm / df.abv
    df["srm_mull_grav"] = df.srm * df.originalGravity
    df["srm_mull_grav_div_abv"] = df.srm * df.originalGravity / df.abv
    return df
