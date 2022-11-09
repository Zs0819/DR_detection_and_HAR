def preprocess(feature, label):
    """Dataset preprocessing move labels from 1-12 to 0-11 for a 12 classes classfication"""
    label -= 1
    return feature, label