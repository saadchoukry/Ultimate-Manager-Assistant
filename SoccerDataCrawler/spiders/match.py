# -*- coding: utf-8 -*-
import scrapy, time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.firefox.options import Options


class MatchSpider(scrapy.Spider):
    name = 'match'
    allowed_domains = ['scoreboard.com']
    start_urls = ['https://www.scoreboard.com/en/soccer/england/premier-league/archive/']
    

    def parse(self, response):
        
        script = "loadN = () =>{if(document.querySelector('a.event__more') != null){document.querySelector('a.event__more').click();} };setInterval(loadN, 3000);"
        ops = Options()
        ops.headless = True
        driver = webdriver.Firefox(options = ops)

        seasons = response.xpath('/html/body/div[4]/div[1]/div/div[1]/div[2]/div[4]/div[3]/div/div[1]/div/a')
        
        for s in seasons[:15]:
            

            driver.get('https://www.scoreboard.com' + s.xpath('@href').get() + 'results/')
            driver.execute_script(script)
            time.sleep(10)         
            
            matchs = driver.find_elements(By.CSS_SELECTOR, '.event__match')
            
            for m in matchs:
                match_url = 'https://www.scoreboard.com/en/match/' + m.get_attribute('id')[4:] + '/#lineups;1'
                
                yield scrapy.Request(url = match_url, callback = self.parse_match)
            
        driver.quit()
            

    def parse_match(self, response):
        
        ops = Options()
        ops.headless = True
        driver = webdriver.Firefox(options = ops)
        
        match = {}
        
        match_url = response.url

        driver.get(match_url + '#lineups;1')
        
        time.sleep(2)

        match['round'] = driver.find_element(By.XPATH, '/html/body/div[2]/div[1]/div[3]/div[1]/span[2]/a').text
        match['datetime'] = driver.find_element(By.XPATH, '//*[@id="utime"]').text
        match['hometeam'] = driver.find_element(By.XPATH, '/html/body/div[2]/div[1]/div[5]/div[2]/div[1]/div[1]/div[2]/div/div/a').text
        match['awayteam'] = driver.find_element(By.XPATH, '/html/body/div[2]/div[1]/div[5]/div[2]/div[1]/div[3]/div[2]/div/div/a').text

        match['hometeam_formation'] = driver.find_element(By.XPATH, '/html/body/div[2]/div[1]/div[5]/div[12]/div[2]/table/tbody/tr/td[1]/b').text
        match['awayteam_formation'] = driver.find_element(By.XPATH, '/html/body/div[2]/div[1]/div[5]/div[12]/div[2]/table/tbody/tr/td[3]/b').text


        match['hometeam_score'] = driver.find_element(By.XPATH, '/html/body/div[2]/div[1]/div[5]/div[2]/div[1]/div[2]/div[1]/span[1]').text
        match['awayteam_score'] = driver.find_element(By.XPATH, '/html/body/div[2]/div[1]/div[5]/div[2]/div[1]/div[2]/div[1]/span[2]/span[2]').text

        hometeam_lineup = []
        awayteam_lineup = []
        hometeam_subs = []
        awayteam_subs = []

        for i in driver.find_elements(By.XPATH, '/html/body/div[2]/div[1]/div[5]/div[12]/div[2]/div[3]/table/tbody/tr'):
            if(i.get_attribute('class') == ''):
                tmp = i.text
            else:
                if(tmp == 'Starting Lineups'):
                    hometeam_lineup.append(i.find_element(By.XPATH, 'td[1]/div[2]').text)
                    awayteam_lineup.append(i.find_element(By.XPATH, 'td[2]/div[2]').text)
                elif(tmp == 'Substitutes'):
                    try:
                        hometeam_subs.append(i.find_element(By.XPATH, 'td[1]/div[2]').text)
                    except:
                        pass
                    try:
                        awayteam_subs.append(i.find_element(By.XPATH, 'td[2]/div[2]').text)
                    except:
                        pass
                    
        try:
            match['hometeam_coach'] = driver.find_element(By.CSS_SELECTOR, '#coaches').find_element(By.XPATH, 'tbody/tr[2]/td[1]/div[2]').text
        except:
            match['hometeam_coach'] = ''
        try:
            match['awayteam_coach'] = driver.find_element(By.CSS_SELECTOR, '#coaches').find_element(By.XPATH, 'tbody/tr[2]/td[2]/div[2]').text
        except:
            match['awayteam_coach'] = ''


        match['hometeam_lineup'] = hometeam_lineup
        match['awayteam_lineup'] = awayteam_lineup
        match['hometeam_subs'] = hometeam_subs
        match['awayteam_subs'] = awayteam_subs
        

        driver.get(match_url + '#match-statistics;1')
        time.sleep(2)
        
        
        for i in driver.find_elements(By.XPATH, '/html/body/div[2]/div[1]/div[5]/div[13]/div[2]/div[4]/div[2]/div'):
            stat_ind = i.find_element(By.XPATH, 'div[1]/div[2]').text
            hometeam_key = 'hometeam_1st_half_' + stat_ind
            awayteam_key = 'awayteam_1st_half_' + stat_ind

            match[hometeam_key] =  i.find_element(By.XPATH, 'div[1]/div[1]').text
            match[awayteam_key] =  i.find_element(By.XPATH, 'div[1]/div[3]').text

        driver.get(match_url + '#match-statistics;2')
        time.sleep(2)

        for i in driver.find_elements(By.XPATH, '/html/body/div[2]/div[1]/div[5]/div[13]/div[2]/div[4]/div[3]/div'):
            stat_ind = i.find_element(By.XPATH, 'div[1]/div[2]').text
            hometeam_key = 'hometeam_2nd_half_' + stat_ind
            awayteam_key = 'awayteam_2nd_half_' + stat_ind
            
            match[hometeam_key] =  i.find_element(By.XPATH, 'div[1]/div[1]').text
            match[awayteam_key] =  i.find_element(By.XPATH, 'div[1]/div[3]').text

        yield match

        driver.quit()
        