
install.packages('ROAuth')

list.of.packages <-c('twitteR','streamR','DBI','RODBC','DBI','streamR','ROAuth','tm')
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]


if(length(new.packages)>0) {
  install.packages(new.packages)
}

lapply(list.of.packages, require, character.only = TRUE)
rm(list.of.packages,new.packages)

#################################################################################

consumer_key <- '0TgapeAfCQOnBdY1Nx1YycjPK'
consumer_secret <- 't1JvqhBKX5b6xO1HeSBYKi8kLBg3tfrqhouIqnTZ0eTgu1ZJ44'
access_token <- '2572138643-afD4rDaMr1QTuz5JvUZYsBUxVRulOaQJSG5RWdG'
access_secret <- 'hzcrINwNViYdBhN5rOBoxaTsphhq9aS8r6rCPkZm85LNr'


requestURL <- "https://api.twitter.com/oauth/request_token"
accessURL <- "https://api.twitter.com/oauth/access_token"
authURL <- "https://api.twitter.com/oauth/authorize"

my_oauth <- OAuthFactory$new(consumerKey=consumer_key,
                             consumerSecret=consumer_secret, requestURL=requestURL,
                             accessURL=accessURL, authURL=authURL)
my_oauth$handshake(cainfo = system.file("CurlSSL", "cacert.pem", package = "RCurl"))

tweets <- filterStream( file.name="", language="en",
                        locations='Munich', timeout=60, oauth=my_oauth )

###########################################################################################

sample<-sampleStream( file.name="",timeout = 600,oauth=my_oauth )

us <- getUser("StockTwits")
us$id

StockTwits<-filterStream(file.name = "",follow = us$id, oauth=my_oauth)
tweets.df  <- parseTweets(sample)
#tweets.df <- parseTweets("tweets_sample.json", simplify = FALSE, verbose = TRUE)

a<-tweets.df[tweets.df$lang == 'en',]$text


temple.words.list<-strsplit(a, '\\W+', perl=TRUE)

temple.words.vector<-unlist(temple.words.list)
temple.words.vector<-temple.words.vector[!grepl(" ",temple.words.vector)]
view(temple.words.vector)
tolower(temple.words.vector)
temple.freq.list<-table(temple.words.vector)
temple.sorted.freq.list<-sort(temple.freq.list, decreasing=TRUE)


temple.sorted.table<-paste(names(temple.sorted.freq.list), temple.sorted.freq.list, sep='\t')
a<-data.frame(sort(temple.freq.list,decreasing = TRUE))

a<- a[a$Freq > 1,]
b<-tolower(a)
