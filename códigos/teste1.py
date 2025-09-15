import pandas as pd
import re
import unicodedata
import os 
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import time

# Download dos recursos do NLTK se necessário
try:
    from nltk.corpus import stopwords
except ImportError:
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
    from nltk.corpus import stopwords

# Definir stopwords em português
STOPWORDS_COMPLETA = set(stopwords.words('portuguese'))

def clean_text(text):
    '''
    Perform stop-words removal and lemmatization
    '''
    text = str(text)
    
    # Converter para minúsculas
    text = text.lower()
    
    # Normalização de caracteres Unicode
    text_normalize = unicodedata.normalize("NFKD", text)
    text = ''.join(
        char for char in text_normalize
        if not unicodedata.combining(char)
    )
    
    # Remover caracteres não ASCII
    text = text.encode('ascii', 'ignore').decode('utf-8')
    
    # Remover URLs, menções e hashtags
    text = re.sub(r'http\S+|https\S+|www\S+|@\S+|#\S+', '', text)
    
    # Remover risadas comuns
    text = re.sub(r'\b(kk|hah|aha|hehe|hihi)\b', '', text)
    
    # Remover pontuação
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    
    # Tokenização e remoção de stopwords
    words = text.split()
    words = [word for word in words if word not in STOPWORDS_COMPLETA and len(word) > 2]
    
    # Lematização
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return " ".join(words)

def load_and_prepare_data(filepath, text_column, label_column):
    """Carrega e prepara os dados de forma eficiente"""
    # Carregar dados
    df_TextLabel = pd.read_csv(filepath)
    
    # Limpar textos
    print("Limpando textos...")
    start_time = time.time()
    
    df_TextLabel['cleaned_text'] = df_TextLabel[text_column].apply(clean_text)
    
    print(f"Textos limpos em {time.time() - start_time:.2f} segundos")
    
    # Filtrar colunas necessárias
    df_TextLabel = df_TextLabel[['cleaned_text', label_column]].copy()
    df_TextLabel.columns = ['text', 'label']
    
    # Remover linhas vazias
    df_TextLabel = df_TextLabel[df_TextLabel['text'].str.len() > 0]
    df_TextLabel = df_TextLabel.dropna()
    
    return df_TextLabel

def split_data(df_TextLabel):
    """Divide os dados em treino e teste"""
    text_var = df_TextLabel['text']
    label_var = df_TextLabel['label']
    return train_test_split(text_var, label_var, test_size=0.2, random_state=42)

def train_and_evaluate_classifiers(datasets_configs, classifiers):
    """
    Treina e avalia múltiplos classificadores em múltiplos datasets de forma eficiente
    """
    results = {}
    
    for dataset_config in datasets_configs:
        dataset_name = dataset_config.get("name", "unknown_dataset")
        print(f"\n{'='*50}")
        print(f"Processando dataset: {dataset_name}")
        print(f"{'='*50}")
        
        try:
            # Carrega e prepara os dados
            start_time = time.time()
            df = load_and_prepare_data(
                dataset_config["filepath"],
                dataset_config["text_column"],
                dataset_config["label_column"]
            )
            load_time = time.time() - start_time
            print(f"Dados carregados em {load_time:.2f}s - {len(df)} amostras")
            
            # Divide os dados
            text_train, text_test, label_train, label_test = split_data(df)
            
            # Vectorização TF-IDF (apenas uma vez por dataset)
            start_time = time.time()
            tfidf_vectorizer = TfidfVectorizer(
                min_df=5, 
                ngram_range=(1, 2),
                max_features=10000
            )
            X_train = tfidf_vectorizer.fit_transform(text_train)
            X_test = tfidf_vectorizer.transform(text_test)
            vectorization_time = time.time() - start_time
            print(f"Vectorização concluída em {vectorization_time:.2f}s - {X_train.shape[1]} features")
            
            # Treina e avalia cada classificador
            for classifier in classifiers:
                classifier_name = classifier.__class__.__name__
                print(f"\nTreinando {classifier_name}...")
                
                # Treina o classificador
                start_time = time.time()
                classifier.fit(X_train, label_train)
                train_time = time.time() - start_time
                
                # Faz predições
                start_time = time.time()
                predictions = classifier.predict(X_test)
                predict_time = time.time() - start_time
                
                # Calcula métricas
                accuracy = accuracy_score(label_test, predictions)
                f1 = f1_score(label_test, predictions, average='weighted')
                
                # Armazena resultados
                if dataset_name not in results:
                    results[dataset_name] = {}
                
                results[dataset_name][classifier_name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'train_time': train_time,
                    'predict_time': predict_time,
                    'model': classifier,
                    'vectorizer': tfidf_vectorizer,
                    'predictions': predictions,
                    'true_labels': label_test.values
                }
                
                print(f"  Acurácia: {accuracy:.4f}")
                print(f"  F1-Score: {f1:.4f}")
                print(f"  Tempo total: {train_time+predict_time:.2f}s")
                
        except Exception as e:
            print(f"Erro ao processar dataset {dataset_name}: {str(e)}")
            continue
    
    return results

# Configurações dos datasets
datasets = [
    {  # HateBR
        "name": "HateBR",
        "filepath": 'Datasets/hateBR/HateBR.csv',
        "text_column": 'comentario',
        "label_column": 'label_final'
    },
    {  # Offcom2
        "name": "Offcom2",
        "filepath": 'Datasets/OffComBR-3/OffComBR3.csv',
        "text_column": 'comentario',
        "label_column": 'label'
    },
    {  # OffcomBR-3
        "name": "OffcomBR-3",
        "filepath": 'Datasets/Offcom2/OffComBR2.csv',
        "text_column": 'mensagem',
        "label_column": 'label'
    },
    {  # OLID-BR 
        "name": "OLID-BR",
        "filepath": 'Datasets/OLID - BR/2019-05-28_portuguese_hate_speech_binary_classification.csv',
        "text_column": 'text',
        "label_column": 'hatespeech_comb'
    },
    {  # BiToLD
        "name": "BiToLD",
        "filepath": 'Datasets/ToLD/ToLD-BR_binario.csv',
        "text_column": 'text',
        "label_column": 'Discurso_de_odio'
    },
]

# Classificadores (instanciados com parâmetros otimizados)
classifiers = [
    MultinomialNB(),
    RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    LinearSVC(random_state=42, max_iter=1000),
    LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
    KNeighborsClassifier(n_jobs=-1)
]

# Executa treinamento e avaliação
if __name__ == "__main__":
    print("Iniciando processo de treinamento e avaliação...")
    results = train_and_evaluate_classifiers(datasets, classifiers)
    
    # Exibe resultados resumidos
    print("\n\n" + "="*60)
    print("RESUMO FINAL DOS RESULTADOS")
    print("="*60)
    
    for dataset_name, dataset_results in results.items():
        print(f"\nDataset: {dataset_name}")
        print("-" * 40)
        for classifier_name, metrics in dataset_results.items():
            print(f"{classifier_name:25} | Acurácia: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f} | Tempo: {metrics['train_time']+metrics['predict_time']:.2f}s")
    
    # Salva resultados em arquivo CSV
    results_df = pd.DataFrame()
    for dataset_name, dataset_results in results.items():
        for classifier_name, metrics in dataset_results.items():
            results_df = pd.concat([results_df, pd.DataFrame({
                'Dataset': [dataset_name],
                'Classificador': [classifier_name],
                'Acurácia': [metrics['accuracy']],
                'F1-Score': [metrics['f1_score']],
                'Tempo_Treino': [metrics['train_time']],
                'Tempo_Predicao': [metrics['predict_time']]
            })], ignore_index=True)
    
    results_df.to_csv('resultados_classificacao.csv', index=False)
    print("\nResultados salvos em 'resultados_classificacao.csv'")