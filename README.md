![Image](https://cdn.discordapp.com/attachments/695016572342894653/740639387221622824/logo.png)
# Ultimate manager assistant

Ultimate manager assistant is a ML powered tool that is designed to help assist football coaches in making better decisions and predict the outcome of a certain game based on parameters characterizing it

## Getting Started

- Clone this repo
- Install the dependencies
```bash
pip freeze > requirements.txt
```
- Scrapping the data the data (ach ay dir)
```bash
scrapy crawl match
```
```bash
scrapy crawl player
```

## Usage

## Usage
Implemented models (3) are:
  - Composure/Defensive awareness recovery:
  - Best position 
  - Match prediction
  
To tune/rerun the training
  ```python
  python dataRecoveryModel.py
  ```

  ```python
  python BestPositionModel.py
  ```
  
  
  ```python
  matchPredictionModel.py
  ```
## Contributing
- Saad Choukry
- Nawfal El Hamdouchi
- Youssef Al Mouatamid
- Yahia Khallouk
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
