SELECT *
FROM main_n_index
INTO OUTFILE '/var/lib/mysql-files/main_n_index.csv'
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n';