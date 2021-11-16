import requests
from bs4 import BeautifulSoup
import unidecode
import math
import re
import nltk
import pandas as pd 
import sys
import PIL.Image as Image
import io 
from urllib.request import urlopen
import os 
import datetime
import time
import pickle
import argparse
import numpy as np
import cv2
from itertools import repeat, count
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor


from nltk.corpus import stopwords

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('maxent_ne_chunker')
# nltk.download('words')

Stemmer = nltk.stem.SnowballStemmer('english')
STOPS = set(stopwords.words('english'))
STOPS.add("")

SAVEPATH = "/mnt/nas2/seungil/crawling/result"


def clean_parenthesis(data : "str"):
    return re.sub("[\[].*?[\]]", "", data)


def clean_text(data : "str") :
    # 1. reference 제거 , \xa0 제거 , \n 제거 
    data = re.sub("[\[].*?[\]]", "", data)
    data = re.sub('\n','',data) 
    data = re.sub('\xa0','',data)

    # 2. only ENG (영문자 이외 문자 공백)
    data = unidecode.unidecode(data) 
    data = re.sub(r'[^a-zA-Z\u00c0-\u00FF0-9\s]', '', data)

    # 3. lower
    # data = data.lower()
    
    return data


def generate_nostop_unigrams(s):
    s = s.split()

    # 불용어 제거
    no_stopwords_list = [word for word in s if not word in STOPS]

    # stemming
    tokens = [Stemmer.stem(word) for word in no_stopwords_list]
    tokens = list(set(tokens))
    return tokens


def my_ner(document):
    capitals = r"((?:[A-Z][a-z]+ ?)+)"
    numbers = r"(?:\+|\-|\$)?\d{1,}(?:\,?\d{3})*(?:\.\d+)?%?"
    ne = set(map(str.strip,re.findall(capitals, document)))
    cd = set(re.findall(numbers, document))
    firstWords = set()
    for sentence in nltk.tokenize.sent_tokenize(document):
        word = sentence.split()[0]
        if not nltk.pos_tag([word])[0][1] in ['NNP','NNPS'] : firstWords.add(word)
    return (ne - firstWords) | cd


def computeTF(wordDict, bow): #각 문단에서 캡션과 겹치는 단어가 발생하는지 1 or 0  #각 문단에서 가지는 단어들 
    tfDict = {}
    bowCount = len(bow)
    if bowCount == 0 : 
      return {}
    else : 
      for word, count in wordDict.items():
          tfDict[word] = count/float(bowCount) # normalizing 이 되는 거 같은데 이게,, 일종의 문단 길이를 반영한다고도 볼 수 있을까??
      return tfDict # 각 문단에서의 캡션에 있는 단어의 frequency 를 나타냄 


def computeIDF(docList): # 각 문단별로 캡션에 있는 단어가 존재하는지 보는 1 or 0의 wordDict들 
    assert(docList)
    idfDict = {}
    N = len(docList)
    idfDict = dict.fromkeys(docList[0].keys(), 0) # 캡션에 존재하는 단어에 대한 Dict 
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1 # 모든 문단들을 보면서 전체 발생 빈도를 체크 
    
    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / (float(val)+1)) # 분모의 +1 은 한번도 발생하지 않는 경우 대비  # 문단전체 / 단어발생비율 >> 단어 발생 횟수가 적을수록 큰 값
        
    return idfDict # 특정 캡션에 대한 IDF score dict 리턴


def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf


def areSubString(captionSet: set, contextSet: set)-> bool:
    for word in captionSet: 
        # if not any(((word in s) or (s in word)) for s in contextSet): return False
        if not any((word in s) for s in contextSet): return False
    return True


def extract_pairs(title, paragraph, imgs, captions, contexts, page_num):
    new_rows = []
    for img, caption in zip(imgs, captions): 
        if not (img and caption):
            # print("AAA") 
            continue # img caption pair 가 True가 아닐 때 
        caption_ner = my_ner(caption) 

        if not caption_ner:
            # print("BBB") 
            continue # NE>1 이면
        valid_contexts = [c for c in contexts if areSubString(caption_ner,my_ner(c[0]))]

        if not valid_contexts:
            # print("CCC") 
            continue #NER을 포함하는 context가 없으면
        uni_cap = generate_nostop_unigrams(caption)

        if len(uni_cap) < 5:
            # print("DDD") 
            continue # caption 의 unigram size가 minimum 을 넘지 못할 때 

        capSet = set(uni_cap) 

        contexts_TF_list = []
        for id in range(len(valid_contexts)):
            contexts_TF_list.append(dict.fromkeys(capSet, 0))
            for word in valid_contexts[id][1]:
                if contexts_TF_list[id].get(word) is not None: contexts_TF_list[id][word] +=1  

        assert(len(contexts_TF_list)==len(valid_contexts))

        computed_TF = [] # 각 문단별로 Term frequency 구하는 과정
        for tf, (_, unigram_context) in zip(contexts_TF_list, valid_contexts):
            computed_TF.append(computeTF(tf, unigram_context))

        idf_score = computeIDF(contexts_TF_list) # 그 소문단에 대한 IDF 계산 

        TFIDF_list = [computeTFIDF(tf, idf_score) for tf in computed_TF]

        max_id, max_score = -1, 0 
        for id, tfidf in enumerate(TFIDF_list) : 
            score = 0
            for val in tfidf.values():
                score += val
            if score > max_score :
                max_score = score 
                max_id = id 

        if max_id == -1:
            # print("EEE") 
            continue  #all negative weights

        context = valid_contexts[max_id][0]
            
        info_dict = {}
        info_dict["title"] = title
        info_dict["paragraph"] = paragraph
        info_dict["caption"] = caption
        info_dict["contexts"] = context
        info_dict["image url"] = img['src']
        info_dict["page num"] = int(page_num)
        info_dict["caption NER"] = caption_ner 
        info_dict["context NER"] = my_ner(context)
        info_dict["TFIDF score"] = max_score 
        # info_dict["caption unigram"] = uni_cap
        # info_dict["context unigram"] = unigram_contexts[max_id] 
        # info_dict["TFIDF"] = TFIDF_list[max_id]

        new_rows.append(info_dict)
    
    return new_rows


def get_partial_pairs(href) : 
  url = "https://en.wikipedia.org" + href

  response = requests.get(url)
  assert (response.status_code == 200)
  
  html = response.text
  soup = BeautifulSoup(html,'html.parser')
  output = soup.select_one('div.mw-parser-output')
  whole_pairs = []
  partial_imgs, partial_captions, partial_contexts = [], [], []
  elements = output.findAll(['div','p','h2'])
  H2 = output.select('h2')

  for elem in elements : 
    if elem.select("span.mw-headline"): # 소제목일 때  
        if H2 : paragraph = H2.pop(0) 
        else : continue
        whole_pairs.append((partial_imgs, partial_captions, partial_contexts, [paragraph.text]))
        partial_imgs, partial_captions, partial_contexts = [], [], []
    elif elem.select("div.thumbinner"): # 이미지를 포함한 div 일 때 
        if elem.select_one("div.thumbinner > div.thumbcaption") :
            partial_imgs.append(elem.select_one("div.thumbinner > a.image > img.thumbimage"))
            partial_captions.append(elem.select_one("div.thumbinner > div.thumbcaption").get_text()) # caption text 만 따옴
    elif elem.name == 'p': # context 일 때
        partial_contexts.append(elem.get_text()) # text 만 따옴 
  return whole_pairs # imgs, captions, context, h2


def get_data_from_page(args, page, csv_dataset, page_num):
    title, href = page
    title = clean_text(title)
    added = 0

    whole_pairs = get_partial_pairs(href) 
    for _, (imgs, captions, contexts, paragraphs) in enumerate(whole_pairs) :
        if not (imgs and captions and contexts and paragraphs): continue

        captions = list(map(clean_text,captions))
        contexts = list(map(clean_text,contexts))
        paragraphs = list(map(clean_text,paragraphs))

        contexts = [(c,generate_nostop_unigrams(c)) for c in contexts] #(context, unigram)의 tuple
        contexts = [c for c in contexts if 15<=len(c[1])<=150] #context 길이 필터링

        new_rows = extract_pairs(title, paragraphs, imgs, captions, contexts, page_num)
        print(title, new_rows)
        for row in new_rows:
            added += 1 
            csv_dataset.append(row ,ignore_index=True)

    if added:
        csv_dataset.to_csv(SAVEPATH, index=True)
        print(f"{[page_num]}:{title} saved")


def get_next_page(url) : 
    response_ = requests.get(url)
    assert (response_.status_code == 200)
    html_ = response_.text 
    soup_ = BeautifulSoup(html_, 'html.parser')

    page_mover = soup_.select('div#mw-content-text > div.mw-allpages-nav > a')

    for item in page_mover : 
        if "Next page" in item.text : return item['href'] 
        else : continue 

    print("Failed to find a next page")
    return False


def check_part_range(part):
    if 1<=part<=10: return part
    raise argparse.ArgumentTypeError("%s is an int part value" % part)


def check_quality_range(quality):
    if quality in ['FA','GA','AA']:
        return quality
    raise argparse.ArgumentTypeError("%s is an int part value" % quality)


def load(part, quality):
    if quality == 'AA':
        with open('./all_pages.pkl','rb') as r : 
            all_urls = pickle.load(r)

        # all_urls are including each urls which contains a bit of article lists 
        # the number of these urls is about 46000 
        # these urls needed to be divided into small parts 
        # basically, we divide it by 10 parts 
        # part = 10 

        partial_num = int(len(all_urls)/part)
        total_parts_list = [ all_urls[partial_num*i : partial_num*(i+1)] if i!= part-1 
                                else all_urls[partial_num*(i):] for i in range(part) ]
    
        for idx, each_part in enumerate(total_parts_list) : 
            with open(f'partial_list_{idx}.pkl', 'wb') as f : 
                pickle.dump(each_part, f)
                
        print(f"each part is downloaded!")

        assert len(total_parts_list) == part 
        print(f"yes, total_parts_list are including {part} lists") 
        
        return total_parts_list 

    elif quality == 'FA':
        featured_articles_wiki = "https://en.wikipedia.org/wiki/Wikipedia:Featured_articles"
        response = requests.get(featured_articles_wiki)

        assert (response.status_code == 200)
        html = response.text 
        soup = BeautifulSoup(html, 'html.parser')
        featured_articles = soup.select('div.mw-parser-output > div.hlist > ul > li')

        pages = []
        for _, article in enumerate(featured_articles) : 
            pages.append((article.get_text(), article.a['href'])) # tuple -> (article title, article href) 
        print(f"all featured article pages are loaded")
        return pages

    elif quality == 'GA':
        def get_good_articles(url) : 
            response_ = requests.get(url)
            assert (response_.status_code == 200)
            html_ = response_.text 
            soup_ = BeautifulSoup(html_, 'html.parser')

            good_lists = soup_.select('div.wp-ga-topic > div.mw-collapsible > div.mw-collapsible-content > p > a')
            return good_lists

        good_articles_wiki = "https://en.wikipedia.org/wiki/Wikipedia:Good_articles/all"
        response = requests.get(good_articles_wiki)

        assert (response.status_code == 200)
        html = response.text 
        soup = BeautifulSoup(html, 'html.parser')

        previous_good_lists = soup.select('div.mw-parser-output > div.wp-ga-topic')[1:] # Contents 제외 
        topics_list = soup.select('div.mw-parser-output > div.wp-ga-summary > div.wp-ga-summary-topics > ul.wp-ga-summary-topics-list > li') # write all the sequences as argument ? 
        good_articles = []
        for each_topic in topics_list : 
            url = "https://en.wikipedia.org" + each_topic.a['href']
            good_articles.extend(get_good_articles(url))

        pages = []
        for _, article in enumerate(good_articles) : 
            pages.append((article.get_text(), article['href']))
        return pages

def get_all_articles(url) : 
    response_ = requests.get(url)
    assert (response_.status_code == 200)
    html_ = response_.text 
    soup_ = BeautifulSoup(html_, 'html.parser')

    all_lists = soup_.select('div#mw-content-text > div.mw-allpages-body > ul > li')
    return all_lists

def crawl(args,iter_pages,csv_dataset,num):
    get_data_from_page(args, iter_pages[num], csv_dataset, num)
    # try : get_data_from_page(args, iter_pages[num], csv_dataset, num)
    # except Exception as e : print(f"Error: {e}, In : {iter_pages[num]}")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--part",
        type=int,
        required = False,
        help="part to crawl during AA"
    )
    parser.add_argument(
        "--quality",
        default= 'AA',
        type=check_quality_range,
        help="part to crawl"
    )
    parser.add_argument(
      "--save",
      type=str,
      default= "/mnt/nas2/seungil/dataset/crawling",
      help="where to save the crawled dataset?"
    )
    parser.add_argument(
      "--load_csv",
      type=str,
      help="load a csv dataset to collect pairs continuously"
    )
    parser.add_argument(
      "--fname",
      type=str,
      default='data.csv',
      help="name of csv file that will be saved"
    )
    parser.add_argument(
      "--aa_pkl",
      type=str,
      default=None, 
      help= 'load a saved pkl to crawl data'
    )
    parser.add_argument(
      "--all_pages_pkl",
      type=str,
      default=None, 
      help = "load a saved all_pages pkl to crawl data "
    )
    args = parser.parse_args()
    
    global SAVEPATH 
    SAVEPATH = args.save + '/' + args.fname

    start = datetime.datetime.now()
    print ("Start: ",start.strftime("%Y-%m-%d %H:%M:%S"))
    # pool = Pool(12) #Multiprocessing with 10 cores

    if args.load_csv : 
      csv_dataset = pd.read_csv(args.load_csv, delimiter = ',', quotechar = '"', sep='\t')  #, engine = 'python'
      start_page =int(csv_dataset.iloc[-1]['page num']+1)
      print(f"crawling process will begin from {start_page}-th article from {args.load_csv}")
    else : 
      csv_dataset = pd.DataFrame()
      start_page = 0

    if args.quality == 'AA': 
        total_parts_list = []

        if args.aa_pkl is not None :
            pkl_name = "/mnt/nas2/seungil/crawling/pages/partial_list_" + args.aa_pkl + ".pkl"
            with open(pkl_name, 'rb') as f : 
                a_list = pickle.load(f)
            total_parts_list.append(a_list)
            count = int(args.aa_pkl)
            print("a pkl length : ",len(total_parts_list[0]))

        elif args.all_pages_pkl is not None :
            pkl_name = "/mnt/nas2/seungil/crawling/all_pages/all_pages_for_partial_list" + args.all_pages_pkl + ".pkl"
            with open(pkl_name, 'rb') as f : 
                all_pages = pickle.load(f) 
            total_parts_list.append(all_pages)
            count = int(args.all_pages_pkl)

        else :
            pages = load(args.part, args.quality)
            total_parts_list = pages 
            count = 0

        while total_parts_list : 
            a_partial_list = total_parts_list.pop(0)  
            all_articles = []
            iter_pages = []
            each_list_len = []
            
            if args.aa_pkl is not None : 
                print("Getting pages ...")
                for each_url in a_partial_list : 
                    #print(f"each url ; {each_url}")
                    all_articles.extend(get_all_articles(each_url))

                for article in all_articles :
                    iter_pages.append((article.get_text(), article.a['href']))
            
                print("We've got all pages for this partition") 

                with open(f"all_pages_for_partial_list{count}.pkl", 'wb') as f : 
                    pickle.dump(iter_pages, f)

                print("all pages for this partition are downloaded") 
            
            if args.all_pages_pkl is not None : 
                iter_pages = a_partial_list
                print(f"A already existed all pages pkl for a partial list is loaded!")
            
            print(f"{count}-th's total pages : {len(iter_pages)}")
            print(f"{count}-th list started !")

            each_list_len.append(len(iter_pages))

            print("Let's crawl them!") 

            # Single Processing
            for num in range(start_page,len(iter_pages)): crawl(args,iter_pages,csv_dataset,num)

            # Multi Processing
            # arguments = zip(repeat(args),repeat(iter_pages),repeat(csv_dataset),range(start_page,len(iter_pages)))
            # pool.starmap(crawl,arguments)

            count += 1
            print(f"{count}-th list in total is done!") 

        print(f"crawling all AA articles is completed! Nice job!")
    
    else : # for FA and GA
        
        pages = load(args.part, args.quality)

        # Single Processing
        # for num in range(start_page,len(pages)): crawl(args,pages,csv_dataset,num)

        # Multi Processing
        arguments = zip(repeat(args),repeat(pages),repeat(csv_dataset),range(start_page,len(pages)))
        pool.starmap(crawl,arguments)

        print(f"All {args.quality} pages are crawled!!!")

    end = datetime.datetime.now()
    print("Elapsed Time:", end-start)

if __name__ == "__main__":
    main()
