[Click here to see the English version](DOCS.md)

## Documentação

### Pré-processamento

Pré-processamento corresponde à obtenção dos dados do banco de dados e métodos auxiliares para organização e limpeza destes dados. O resultado pode ser encontrado na pasta [data]() e no arquivo paris.txt.

**get_data_from_database_and_save_to_file(query, filename, isJson=True)**
> Faz a leitura dos dados do banco de dados e os salva em um arquivo. Este método possui os dados de conexão com o banco de dados. Então executa a query que será utilizada na consulta ao banco de dados, essa query foi recebida como argumento do método. Logo, salva no arquivo os dados resultantes da consulta. O nome do arquivo em que os dados são salvos, foi recebido por argumento da função, além da determinação se os dados estão ou não no formato JavaScript Object Notation (JSON). Esta informação é utilizada para o correto salvamento dos dados no arquivo. O retorno é o nome do arquivo em que os dados foram salvos, sem a especificação da extensão.  

**get_manual_classified_tweets_from_database(filename)**
> Define a query para leitura dos 1.200 tweets que foram classificados manualmente, além de especificar o nome do arquivo onde os dados serão salvos. Então o método get_data_from_database_and_save_to_file é chamado, e a query e nome do arquivo definidos são passados como parâmetros do método.  

**get_manual_classification_from_database(filename)**
> Define a query para leitura da classificação manual feita sobre 1.200 tweets, além de especificar o nome do arquivo onde os dados serão salvos. Então o método get_data_from_database_and_save_to_file é chamado, e a query e nome do arquivo definidos são passados como parâmetros do método.  

**get_all_tweets_from_database(filename)**
> Define a query para leitura de todos os tweetsda base de dados, além de especificar o nome do arquivo onde os dados serão salvos. Então o método get_data_from_database_and_save_to_file é chamado, e a query e nome do arquivo definidos são passados como parâmetros do método.  

**clean_data(input_filename, output_filename, isJson=True)**
> Faz a leitura do arquivo de texto, alterações no conteúdo lido e exportação do conteúdo alterado para outro arquivo de texto. As alterações são baseadas no erros mais comuns enfrentados no uso de dados no formato JSON, identificados previamente, como a presença de aspas como conteúdo de um campo, por exemplo, no campo source, que é composto por uma tag HyperText Markup Language (HTML) <a>, que representa um link. O erro ocorre porque as aspas delimitam o fim do campo em local incorreto, e o conteúdo restante não será considerado nem chave nem conteúdo do JSON, apontando-o como um JSON inválido.  

### Detecção de Spammer

**manual_classification(f_tweets, f_classifications)**
> Cruza os dados dos tweets e sua classificação manual, retornando uma lista que associa um número de identificação do usuário (user ID) com a classificação (“spam” ou “não spam”). É possível que sejam feitas classificações duplicadas.

**get_criteria_features_by_user(f_tweets)**
> Utilizando-se as características de cada tweet e baseando-se nos nove critérios propostos por El Azab et al. (2016), para cada tweet, é gerado um vetor com nove valores booleanos, sendo cada valor, uma indicação se a característica atende a cada critério. Com “0” (zero) representando que a característica não atende ao critério e “1” (um) representando que a característica atende ao critério. O resultado é uma matriz 2xN, onde N é o número de tweets classificados e, para cada linha são mostrados o número de identificação do usuário (user ID) e o vetor booleano.

**get_prob_based_on_criteria(features)**
> Ao receber o vetor booleano de características, este método calcula a probabilidade de cada usuário ser spammer, baseando-se nos pesos apresentados no trabalho de El Azab et al. (2016). O retorno é uma lista 2xN, onde N é o número de tweets classificados e, para cada linha são mostrados o número de identificação do usuário (user ID) e a probabilidade de o usuário ser spammer.

**remove_duplicates_prob_criteria(users_prob_criteria)**
> Ao receber uma lista com o número de identificação do usuário (user ID) e a probabilidade de o usuário ser spammer, este método retira usuários duplicados da lista, atribuindo a ele a maior probabilidade dentre todas as probabilidades atribuídas a ele na lista. O retorno é uma lista 2xN, onde N é o número de tweets classificados e, para cada linha são mostrados o número de identificação do usuário (user ID) e a probabilidade de o usuário ser spammer, onde cada usuário ocorre apenas uma única vez na lista.

**criteria_classification(users_prob_criteria)**
> A partir de uma lista com o número de identificação do usuário e a probabilidade deste usuário ser spammer, classifica o usuário como
“spammer” ou “não spammer”, a partir do limiar 0.991, que determina a classificação. 

**trains_decision_tree(features_list, classifications)**
> A partir de uma lista com o número de identificação do
usuário (user ID) e seu vetor booleano de características e uma lista com o número de identificação do usuário (user ID) e sua classificação manual, é determinada a classificação de cada lista booleana, cruzando-se os dados baseando-se na identificação do usuário. Ambas, classificação e lista booleana, são utilizadas para gerar um classificador baseado em árvore de decisão. O retorno é o classificador.

**trains_bernoulli(features_list, classifications)**
> A partir de uma lista com o número de identificação do usuário (user ID) e seu vetor booleano de características e uma lista com o número de identificação do usuário (user ID) e sua classificação manual, é determinada a classificação de cada lista booleana, cruzando-se os dados baseando-se na identificação do usuário. Ambas, classificação e lista booleana, são utilizadas para gerar um classificador baseado em Naïve Bayes Bernoulli Multivariante. O retorno é o classificador.

**trains_multinomial(features_list, classifications)**
> A partir de uma lista com o número de identificação do usuário (user ID) e seu vetor booleano de características e uma lista com o número de identificação do usuário (user ID) e sua classificação manual, é determinada a classificação de cada lista booleana, cruzando-se os dados baseando-se na identificação do usuário. Ambas, classificação e lista booleana, são utilizadas para gerar um classificador baseado em Naïve Bayes Multinomial. O retorno é o classificador.

**classification(clf, data)**
> Recebe um classificador e os dados, que são tweets no formato JSON, e realiza a classificação dos dados utilizando-se o classificador. O retorno é uma lista com a classificação.

**final_classification(criteria, bernoulli, decision_tree, multinomial)**
> A partir de quatro classificações, baseado nos critérios propostos por El Azab et al. (2016), Naïve Bayes Bernoulli, Naïve Bayes Multinomial e Decision Tree, determina qual a classificação final do tweet: “spam” ou “não spam”.

**cross_validation_10_fold(clf, features_list, classifications)**
> Realiza a validação cruzada 10-fold a partir de um classificador, uma lista com vetores de características booleanos e a classificação manual, recebidos como argumentos do método. O retorno é uma porcentagem que representa a precisão do classificador.

**confusion_matrix_(true, predicted, title)**
> A partir da classificação feita por um classificador e pela classificação feita manualmente, os resultados dessas classificações são cruzados e uma matriz de confusão é gerada. A matriz de confusão mostra a correspondência entre ambas classificações.

**criteria_accuracy(predicted, true)**
> A partir da classificação manual e da classificação feita pelos critérios do artigo de El Azab et al. (2016), é calculada a precisão da classificação feita a partir destes critérios.

### Análise de Sentimentos

**criteria_analysis(user_id, tweet, f_output)**
> Para cada tweet, se o usuário não for provável spammer, o método sentiment_analysis é chamado, que verifica quais sentimentos estão presentes no tweet.

**sentiment_analysis(dict_classif, f_data, f_output)**
> A partir de um arquivo de texto, faz a leitura do texto de cada tweet e verifica se alguma palavra da lista de sentimentos está presente nele. Estando presente, o sentimento associado à palavra é associado ao número de identificação do usuário (user ID). O número de identificação do usuário (user ID) e os sentimentos são salvos em um arquivo de texto.

### Características dos Dados

**plot_and_save_graphics**
> Extrai a data e hora em que cada tweet foi publicado, calcula o número de tweets publicados por usuário, informa quantos usuários únicos há na amostra e quantos tweets foram publicados por dia. Então salva no arquivo um gráfico que mostra quantos usuários publicaram uma quantidade de tweets (exemplo: 30 usuários publicaram 2 tweets cada). Também salva no arquivo um gráfico com o número de tweets por dia da semana.

**display_general_characteristics**
> Exibe o número total de tweets, o número de usuários únicos, a média de tweets por usuário e o período em que os tweets foram publicados. Todas essas informações são salvas em arquivo.

**classification_results**
> Exibe de forma formatada quantos tweets foram classificados como “spam” e “não spam” para uma lista de tweets e classificações.
