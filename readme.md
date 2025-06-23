# 🛳 Titanic - Machine Learning do Desastre

Este repositório contém um resumo em português das informações essenciais para participar da clássica competição do Titanic na Kaggle.

---

## 🎯 Objetivo do Desafio

O naufrágio do Titanic é um dos acidentes marítimos mais famosos da história.

Em 15 de abril de 1912, durante sua viagem inaugural, o "inafundável" RMS Titanic colidiu com um iceberg. Como não havia botes salva-vidas suficientes, 1502 das 2224 pessoas a bordo morreram.

Embora a sorte tenha influenciado, certos grupos de pessoas tinham **mais chances de sobreviver**.

O desafio é criar um **modelo preditivo de machine learning** que responda:

> "Que tipo de pessoa tinha mais chance de sobreviver?"

Para isso, usaremos dados como nome, idade, gênero e classe social dos passageiros.

---

## 📁 Conjuntos de Dados

Você receberá dois arquivos principais:

### `train.csv`

- Dados de **891 passageiros**
- Contém a coluna `Survived` (1 = sobreviveu, 0 = morreu)
- Usado para **treinar o modelo**

### `test.csv`

- Dados de **418 passageiros**
- Sem a coluna `Survived`
- Você precisa **prever se cada passageiro sobreviveu**

Explore os dados na aba **Data** da competição.
## 🔧 Como usar este repositório

1. Instale as dependências (pandas, scikit-learn, numpy).
2. Execute `python main.py` para gerar o arquivo `submission.csv`.


---

## ⚙ Como Funciona a Competição

### 1. 📝 Participe

- Clique em "Join Competition"
- Aceite as regras
- Baixe os dados

### 2. 🧪 Modele

- Use os dados localmente ou com **Kaggle Notebooks** (Jupyter com GPUs gratuitas)

### 3. 📤 Submeta

- Envie sua previsão no formato `.csv`
- Receba uma pontuação com base na acurácia

### 4. 📊 Confira o Ranking

- Veja sua posição no leaderboard

### 5. 🔁 Melhore

- Participe dos fóruns, leia insights, aprenda com os notebooks da comunidade

---

## 📄 Formato do Arquivo de Submissão

O arquivo `.csv` deve conter **exatamente 418 linhas + cabeçalho**, com as colunas:

```csv
PassengerId,Survived
892,0
893,1
...
```

- **PassengerId**: pode estar em qualquer ordem
- **Survived**: 0 = morreu, 1 = sobreviveu

Erros comuns:

- Colunas extras
- Número incorreto de linhas

---

## 💬 Onde Buscar Ajuda

- Use o [**fórum da competição**](https://www.kaggle.com/c/titanic/discussion)
- A Kaggle **não tem suporte direto para seu código**, mas a comunidade é muito ativa
- Compartilhe o que você aprende e os outros também compartilharão com você

---

## 📚 Kaggle Notebooks

O ambiente **Kaggle Notebooks** é uma forma rápida e gratuita de:

- Executar seus códigos na nuvem
- Usar GPUs
- Testar e compartilhar seus modelos

Confira os [Notebooks da competição aqui](https://www.kaggle.com/c/titanic/code) para se inspirar.

---

## 🚀 Pronto para Começar?

1. Cadastre-se na competição [nesta página](https://www.kaggle.com/c/titanic)
2. Siga o [tutorial da Alexis Cook](https://www.kaggle.com/code/alexisbcook/titanic-tutorial)
3. Crie seu primeiro modelo
4. Envie e veja sua pontuação

Boa sorte e bom aprendizado! 💡

