AA 

conda activate seungil

1) all_pages_for_a_partial_list pkl
	python crawler/crawler.py --fname AA0.csv --all_pages_pkl 0 --load_csv /mnt/nas2/seungil/crawling/result/AA0.csv
	python crawler/crawler.py --fname AA1.csv --all_pages_pkl 1 --load_csv /mnt/nas2/seungil/crawling/result/AA1.csv
	

2) partial_page pkl 활용할 경우 

	python crawler/crawler.py --fname AA0.csv --aa_pkl 0
	python crawler/crawler.py --fname AA1.csv --aa_pkl 1
	python crawler/crawler.py --fname AA2.csv --aa_pkl 2
	python crawler/crawler.py --fname AA3.csv --aa_pkl 3
	python crawler/crawler.py --fname AA4.csv --aa_pkl 4
	python crawler/crawler.py --fname AA5.csv --aa_pkl 5
	python crawler/crawler.py --fname AA6.csv --aa_pkl 6
	python crawler/crawler.py --fname AA7.csv --aa_pkl 7
	python crawler/crawler.py --fname AA8.csv --aa_pkl 8
	python crawler/crawler.py --fname AA9.csv --aa_pkl 9