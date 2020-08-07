![Image](https://cdn.discordapp.com/attachments/695016572342894653/740639387221622824/logo.png)
# Ultimate manager assistant

Ultimate manager assistant is a ML powered tool that is designed to help assist football coaches in making better decisions and predict the outcome of a certain game based on parameters characterizing it

## Getting Started

- Cloning this repo
- Installing the dependencies
```bash
pip freeze > requirements.txt
```
- Scrapping the data (or download it from this [link](https://drive.google.com/drive/u/0/folders/1auaMhuaCTkLGLXwcmeB8Zn9gWGfAfkgv))
```bash
scrapy crawl match -o PL_PLYRS.json -t json
```
```bash
scrapy crawl player -o PL_RESULT.json -t json
```

## Usage
After downloading the datasets and moving them to the root of the project, you will need to dispatch the one related to players:
```bash
python playersDispatcher.py
python matchesPlayersLoader.py
```
New files will be added to "dispatchedPlayers" directory

# Models
Implemented models (3) are:
  - Composure/Defensive awareness recovery:
  - Best position 
  - Match prediction
  
To tune/rerun the training, edit and run:
  ```bash
  python dataRecoveryModel.py
  ```

  ```bash
  python BestPositionModel.py
  ```
  
  
  ```bash
  python matchPredictionModel.py
  ```
## Contributing
- Saad Choukry
- Nawfal El Hamdouchi
- Youssef Al Mouatamid
- Yahia Khallouk\
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
