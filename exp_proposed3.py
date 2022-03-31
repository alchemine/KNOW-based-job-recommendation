from analysis_tools.common import *

np.random.seed(RANDOM_STATE)


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

train_full_data = pd.read_csv(join(PATH.TRAIN, 'KNOW_2017.csv'), index_col=0)
X_test          = pd.read_csv(join(PATH.TEST, 'KNOW_2017_test.csv'), index_col=0)
target          = 'knowcode'
nlp_cols        = ['bq4_1a', 'bq4_1b', 'bq4_1c', 'bq5_2', 'bq19_1', 'bq30', 'bq31', 'bq32', 'bq33', 'bq34', 'bq38_1']

train_full_ratio   = 0.3
_, train_full_data = train_test_split(train_full_data, stratify=train_full_data[target], test_size=train_full_ratio)

train_full_data_ = copy(train_full_data)
X_train_full = train_full_data.drop(columns=target)
y_train_full = train_full_data[target]

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, stratify=y_train_full)

oh_enc     = OneHotEncoder(sparse=False)
y_train_oh = oh_enc.fit_transform(y_train[:, None])
y_val_oh   = oh_enc.transform(y_val[:, None])

print("- Train:", X_train.shape, y_train.shape, y_train_oh.shape)
print("- Val:", X_val.shape, y_val.shape, y_val_oh.shape)
print("- Test:", X_test.shape)


from analysis_tools.preprocessing import *

from googletrans import Translator
import torch
from transformers import BertTokenizer, BertModel
from sklearn.decomposition import PCA


class Preprocessor3:
    def __init__(self, nlp_cols, selected_nlp_cols, translate=False):
        self.nlp_cols = nlp_cols
        self.selected_nlp_cols = selected_nlp_cols
        self.translate = translate
        self.imputer = get_imputer()
        if translate:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.model = BertModel.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
        self.dim_emb = 768
        self.pcas = {}

    def fit_transform(self, X, y=None):
        X_imputed = self.imputer.fit_transform(X).astype(str)
        X_non_nlp = X_imputed.drop(columns=self.nlp_cols)
        X_nlp = X_imputed[self.selected_nlp_cols]
        self.dics = self._get_vector_dics(X_nlp)
        X_nlp = self._allocate_vector(X_nlp, self.dics)
        return X_non_nlp.join(X_nlp)

    def transform(self, X, y=None):
        X_imputed = self.imputer.transform(X).astype(str)
        X_rst = self._allocate_vector(X_imputed, self.dics)
        return X_rst

    def _get_vector_dics(self, X):
        dics = {}
        for col in tqdm(X):
            texts = X[col].unique()
            tasks = [delayed(self._text2vector)(text, self.tokenizer, self.model, self.translate) for text in texts]
            with ProgressBar():
                vecs = compute(*tasks)
            self.pcas[col] = PCA(n_components=0.8, random_state=RANDOM_STATE)
            vecs_pca = self.pcas[col].fit_transform(vecs)
            dics[col] = dict(zip(texts, vecs_pca))
        return dics

    @staticmethod
    def _text2vector(text, tokenizer, model, translate):
        def text2input(text):
            marked_text = f"[CLS] {text} [SEP]"
            tokenized_text = tokenizer.tokenize(marked_text)
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
            segments_ids = [1] * len(tokenized_text)
            return torch.tensor([indexed_tokens]), torch.tensor([segments_ids])

        if translate:
            trans = Translator()
            text = trans.translate(text, target='en').text
        with torch.no_grad():
            outputs = model(*text2input(text))
        last_hidden_state, pooler_output, hidden_states = outputs.values()  # use only hidden_states
        token_vecs = hidden_states[-2][0]  # second to last
        return token_vecs.mean(axis=0).numpy()

    def _allocate_vector(self, X, dics):
        rst = pd.DataFrame(index=X.index)
        for col, dic in dics.items():
            f = X[col]
            emb_cols = [f"{col}_{i}" for i in range(self.pcas[col].n_components_)]
            texts = X[col].unique()

            unknown_texts = [text for text in texts if text not in dic]
            if unknown_texts:
                tasks = [delayed(self._text2vector)(text, self.tokenizer, self.model, self.translate) for text in
                         unknown_texts]
                with ProgressBar():
                    unknown_vecs = compute(*tasks)
                unknown_vecs_pca = self.pcas[col].transform(unknown_vecs)
                dic.update(dict(zip(unknown_texts, unknown_vecs_pca)))

            for text in tqdm(X[col].unique()):
                idxs = f[f == text].index
                vec = dic[text]
                rst.at[idxs, emb_cols] = vec
        return rst

preprocessor3 = Preprocessor3(nlp_cols=nlp_cols, selected_nlp_cols=['bq30'])
x = preprocessor3.fit_transform(X_train)