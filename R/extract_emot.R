

findEmoticons <- function(str){
  if(!exists("emDict") | !exists("emDict.rCode")){
    emDict <- read.csv2("emDict.csv")
    emDict.rCode <- as.character(emDict[[4]])
  }
  
  ret <- sapply(emDict.rCode,grepl,str)
  ret <- paste("<",as.character(emDict[ret,1]),">",sep = "",collapse="")
  ret[ret=="<>"] = ""
  #ret <- as.character(emDict[ret,1])
  return(unlist(ret))
}


test <- lapply(X = tweets2,FUN = findEmoticons)

d<-data.frame(tweets2)
d$emot <- unlist(test)


write.csv2(d,file = "tweets2.csv")
