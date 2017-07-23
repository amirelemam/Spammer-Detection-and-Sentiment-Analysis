[Clique aqui para ver a versão em Português](README_PT.md)

## Spammer Detection and Sentiment Analysis

### SPAM Detection


| Attribute                                        | Weight |
| :----------------------------------------------: |:------:|
| The account has at least 30 followers            | 0.53   |
| The account has been geo-localized               | 0.85   |
| It has been included in another user's favorites | 0.85   |
| It has used a hashtag in at least one tweet      | 0.85   |
| It has logged into Twitter using an iPhone       | 0.917  |
| A mention by Twitter user                        | 1      |
| It has written at least 50 tweets                | 0.01   |
| It has been included in another user's list      | 0.45   |
| (2*number of followers) _ (number of friends)    | 0.5    |
| User has at least one Favorite list              | 0.17   |

### License

This project is licensed under the [MIT License](LICENSE)