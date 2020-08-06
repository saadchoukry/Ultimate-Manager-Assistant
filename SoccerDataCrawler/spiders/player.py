# -*- coding: utf-8 -*-
import scrapy


class PlayerSpider(scrapy.Spider):
    name = 'player'
    allowed_domains = ['sofifa.com']
    start_urls = ['https://sofifa.com/players?lg=13']


    def parse(self, response):
        seasons = response.xpath('/html/body/header/div[2]/div/h2/div[1]/div/a/@href').getall()

        for season_url in seasons:
            yield scrapy.Request(url = 'https://sofifa.com'+ season_url, callback = self.parse_season)


    def parse_season(self, response):
        dates_links = response.xpath('/html/body/header/div[2]/div/h2/div[2]/div/a')

        for date_link in dates_links:
            date = date_link.xpath('text()').get()
            date_url = date_link.xpath('@href').get()

            yield scrapy.Request(url = 'https://sofifa.com'+ date_url, callback = self.parse_period, meta = {'Date' : date})



    def parse_period(self, response):

        players = response.xpath('/html/body/div[1]/div/div/div[1]/table/tbody/tr/td[2]/a[1]/@href').getall()

        for player_url in players:
            
            yield scrapy.Request(url = 'https://sofifa.com'+ player_url, callback = self.parse_player, meta = {'Date' : response.meta.get('Date')})

            tmp = response.xpath('/html/body/div[1]/div/div/div[1]/div[1]/a')
            if len(tmp) == 2:
                next_page_url = tmp[-1].xpath('@href').get()
            else:
                next_page_url = tmp.xpath('@href').get()

            if next_page_url:
                yield scrapy.Request(url = 'https://sofifa.com' + next_page_url, callback = self.parse_period, meta = {'Date' : response.meta.get('Date')})
        
    def parse_player(self, response):
        
        player = {}

        player['date'] = response.meta.get('Date')

        player['id'] = response.xpath('/html/body/div[2]/div/div/div[1]/div/div[1]/div/div/h1/text()').get()
        player['gen_info'] = (response.xpath('/html/body/div[2]/div/div/div[1]/div/div[1]/div/div/div/text()').getall()[-1]).replace('"', '')
        player['country'] = response.xpath('/html/body/div[2]/div/div/div[1]/div[1]/div[1]/div/div/div/a/@title').get()
        player['pos'] = response.xpath('/html/body/div[2]/div/div/div[1]/div/div[1]/div/div/div/span/text()').get()
        player['team'] = response.xpath('/html/body/div[2]/div/div/div[1]/div[1]/div[2]/div/div[3]/div/h5/a/text()').get()
        

        attributes = (response.xpath('/html/body/div[2]/div/div/div[1]/div/div')[3:-2]).xpath('div/ul/li')

        for attr in attributes:
            if attr.xpath('text()').get().strip() == '':
                player[attr.xpath('span')[-1].xpath('text()').get()] = attr.xpath('span')[0].xpath('text()').get()
            else:
                player[attr.xpath('text()').getall()[-1].strip()] = attr.xpath('span/text()').get()
        
        yield player
