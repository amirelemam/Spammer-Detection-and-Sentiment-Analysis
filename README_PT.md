[Click here to see the English version](README.md)

## Detecção de Spammer e Análise de Sentimentos

### Descrição

Este projeto consiste na análise de um grupo de 7.189.160 __tweets__, que foram filtrados a partir de hashtags relacionadas aos ataques terroristas em Paris, em 13 de novembro de 2015. 

O objetivo principal é identificar __spammers__ através das características do usuário, que são obtidas nos metadados do __tweet__. O objetivo secundário é identificar sentimentos experimentados pelos usuários durante a postagem de seus __tweets__.

Foi escolhida a análise dos metadados e não do texto do tweet, porque este método é tido na literatura como a segunda melhor maneira de detecção de spam, com eficácia superada apenas pelo estudo de relacionamentos de usuários, através de grafos.

A opção por não fazer o estudo de relacionamentos de usuários deve-se ao seu alto custo computacional e utilização de técnicas avançadas, sendo que este estudo tem por objetivo ser uma análise introdutória.

### Detecção de SPAM

Da base, 1.200 tweets foram coletados de forma aleatória da amostra e anotados manualmente como "SPAM" ou "NAO SPAM".

Estes tweets foram usados para treinar os três classificadores, criados a partir dos algorítmos Naïve Bayes Bernoulli, Naïve Bayes Multinomial e Decision Tree. 

O quarto classificador foi feito baseando-se nos critérios propostos por El Azab et al. (2016), que dá um peso a cada característica do tweet, conforme tabela abaixo:

```

```

Com os quatro classificadores treinados, o tweet era indicado como SPAM caso a maioria dos algorítmos apontasse como tal. Caso houvesse empate, a classificação feita pelos critérios de El Azab et al. (2016) tinha preferência.

### Análise de Sentimentos

A análise de sentimentos foi feita baseando-se na lista de Benevenuto et al. (2010), onde uma lista de palavras era associada a um sentimento, a lista é a seguinte:

```

```

Caso a palavra se encontrasse no tweet, o tweet era "marcado" com aquele sentimento. Um tweet poderia expressar múltiplos sentimentos.

### Métodos

Segue abaixo a lista com o nome dos métodos e seu papel no código:

```

```
### Arquivos

**remove_training**: Retira do arquivo com os dados os tweets usados no treinamento. 
**quebralinhas**: Quebra o arquivo com os dados em 15 arquivos, com 500 mil tweets cada, este número foi definido baseando-se no tamanho médio que o arquivo de 500 mil tweets tinha. Deveria ser um tamanho adequado para o servidor.
**bash**: Gerencia a fila de arquivos, fazendo com que o processamento de cada arquivo seja feito de forma sequencial.
**spammer_detection_and_sentiment_analysis**: Faz a análise de cada tweet para detectar spam e fazer análise de sentimentos. 
**paris.txt** (externo): Base de dados com 7.189.160 tweets, relacionados aos ataques terroristas de 13 de novembro de 2015, em Paris, na França. A base de dados completa pode ser obtida [aqui](), no [Mirror 1]() e no [Mirror 2]() (46GB). 

### Resultado

A classificação dos algorítmos em relação aos tweets serem ou não SPAM indicou alto índice de SPAM, conforme tabela abaixo:

```
Da base de 7.189.160 tweets, foram retirados os 1.200 tweets utilizados para treinamento e 1.516 tweets com conteúdo inválido para o algoritmo, restando 7.186.444 tweets, que foram classificados a partir dos classificadores. O classificador Naïve Bayes Bernoulli classificou 1.366.250 tweets (19%) dos tweets como “spam” e 5.820.194 tweets (81%) como “não spam”, o classificador Naïve Bayes Multinomial classificou 491.120 tweets (6,8%) como “spam” e 6.695.324 tweets (93,2%) como “não spam”, o classificador Árvore de Decisão classificou 1.262.236 tweets (17,6%) como “spam” e 5.924.208 tweets (82,4%) como “não spam” e o classificador a partir dos critérios de El Azab et al. (2016) classificou 2.043.035 tweets (28,4%) como “spam” e 5.143.409 tweets (71,6%) como “não spam”. Baseando-se nos critérios apresentados no Procedimento Metodológico, a classificação final foi de 657.476 tweets (9,1%) como “spam” e 6.528.968 tweets (90,9%) como “não spam”.
```

A análise de sentimentos identificou todos os sentimentos contidos na tabela de Benevenuto et al. (2010), o que confirma que sensações ruins relacionadas aos ataques foram geradas nas populações francesa e mundial.

### Problemas

Vários problemas foram identificados após o fim do projeto e podem ser encontrados na seção `issues`. Contribuições são muito bem-vindas.

### Licença

Estes arquivos estão licenciados sob a [Licença MIT](LICENSE)
