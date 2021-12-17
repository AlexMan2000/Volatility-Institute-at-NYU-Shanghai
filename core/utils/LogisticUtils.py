from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def tokenNumberNormalizer(tokens):
    """ 将所有数字标记映射为一个占位符（Placeholder）。
    对于许多实际应用场景来说，以数字开头的tokens不是很有用，
    但这样tokens的存在也有一定相关性。 通过将所有数字都表示成同一个符号，可以达到降维的目的。
    """
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)


# Basic Model (基于TF-IDF Vectorizer和多分类逻辑回归的实现)
class NLPNormalizeVectorizer(TfidfVectorizer):
  def build_tokenizer(self):
    tokenize = super().build_tokenizer()
    #返回一个匿名函数，后续可以传递参数doc(string类型)调用它
    return lambda doc: list(tokenNumberNormalizer(tokenize(doc)))