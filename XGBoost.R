
###############################
## XGBoost for Classification #
###############################

#Please note, if you want to save time and only want too see the predictions from the models and the corresponding graphs, you can load XGBoost_Exercise.RData 
#In this way, you will not need to load the speech data, create the tokens and train the model

#load('XGBoost_Exercise.RData')

set.seed(221)
##
# Load packages
  install.packages('pacman')
  library(pacman)
  pacman::p_load(reshape2, caret, quanteda, zoo, lubridate, tidyverse, tidytext, Hmisc, car, manifestoR, dplyr, stringr, stringi, readr, xgboost, ggplot2, utils, quanteda.textstats, summarytools, parallel, performanceEstimation)

##
# Functions
  umlauts <- c("ä", "ö", "ü", "Ä", "Ö", "Ü","ß","ğ")
  umlauts_suggested_change <- c("ae", "oe", "ue", "Ae", "Oe", "Ue","ss","g")
  
  umlaut <- function(x) {
    umlaut1 <- stri_replace_all_fixed(x,umlauts, 
                                      umlauts_suggested_change,
                                      vectorize_all = FALSE)
    return(umlaut1)
  }

##
# Load Data
  download.file(url = 'https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/L4OAKN/PCYUNY', 'Corp_Bundestag_V2.rds', method = 'libcurl', cacheOK = TRUE, )

  Corp_Bundestag_V2 <- readRDS("Corp_Bundestag_V2.rds")
  
  #If this does not work, download the dataset from 'https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/L4OAKN' and save it in the working directory
  #as 'Corp_Bundestag_V2.rds' and load with load('Corp_Bundestag_V2.rds')
  
##
# Pre-processing

  #Select speeches from a particular parliament. Here we take half of the 14th Bundestag from 17 October 2002 until 17 October 2002
  Corp_Bundestag_V2$date <- as.Date(Corp_Bundestag_V2$date)
  
  Corp_Bundestag_V2 <- subset(Corp_Bundestag_V2, Corp_Bundestag_V2$date >= '2000-10-17' & Corp_Bundestag_V2$date <= '2002-10-17')
  
  #Remove special characters
  Corp_Bundestag_V2$text <- umlaut(Corp_Bundestag_V2$text)
  
  #subset speeches not from the Chair
  Corp_Bundestag_V2 <- Corp_Bundestag_V2[!(Corp_Bundestag_V2$chair=="TRUE"),]
  
  #remove speeches from independents
  Corp_Bundestag_V2 <- subset(Corp_Bundestag_V2, Corp_Bundestag_V2$party != "independent")
  
  #symbols and interruptions
  Corp_Bundestag_V2$text <- sub("^\\s*<U\\+\\w+>\\s*", "", Corp_Bundestag_V2$text)
  Corp_Bundestag_V2$text <- str_remove_all(Corp_Bundestag_V2$text, "\\s*\\([^\\)]+\\)")
  Corp_Bundestag_V2$text <- str_remove_all(Corp_Bundestag_V2$text, "_")
  Corp_Bundestag_V2$text <- gsub("[0-9] \\.","",Corp_Bundestag_V2$text)
  
  #Create Tokens with ngrams of 1:4, a Document Frequency Matrix and Remove Stopwords, Numbers and Punctuation
  toks <- Corp_Bundestag_V2 %>% corpus(text_field = "text") %>% tokens(remove_punct = TRUE, remove_symbols = TRUE, include_docvars = TRUE) %>% tokens_select(pattern = stopwords("de"), selection = "remove", min_nchar = 2)
  
  #find collocations (not strictly necessary for the analysis)
  col <- toks %>% 
    tokens_select(pattern = "^[A-Z]", valuetype = "regex", 
                  case_insensitive = FALSE, padding = TRUE) %>% 
    textstat_collocations(min_count = 5, tolower = FALSE)

  col2 <- toks %>% textstat_collocations(size = 2, min_count = 10, tolower = FALSE) 
  
  col3 <- toks %>% textstat_collocations(size = 3, min_count = 25, tolower = FALSE)
  
  col4 <- toks %>% textstat_collocations(size = 4, min_count = 5, tolower = FALSE)
  
  col5 <- toks %>% textstat_collocations(size = 5, min_count = 25, tolower = FALSE)
  
  col6 <- toks %>% textstat_collocations(size = 6, min_count = 25, tolower = FALSE) 
  
  #Combine collocation datasets that are useful
  colcombined <- rbind(subset(col, col$z > 15), subset(col2, col2$z > 15), subset(col3, col3$z > 15), subset(col4, col4$z > 15), subset(col5, col5$z > 15), 
                       subset(col6, col6$z > 15))
  
  comp_toks <- tokens_compound(toks, pattern = colcombined, case_insensitive = FALSE, join = TRUE)
  
  #Create a document frequency matrix (dfm)
  Corp_Bundestag_V2_dfm <- dfm(comp_toks)
  
  #Cut off terms that are used less than 50 times and a maximum of 300 times in minimal 10 speeches and apply a TF-IDF tranformation, so that
  #there are more speeches than vocabulary
  Corp_Bundestag_V2_dfm <- dfm_trim(Corp_Bundestag_V2_dfm, min_termfreq = 50, max_termfreq = 300, min_docfreq = 10) %>% dfm_tfidf()

  ################################################
  # Boosting Cross-validation -> binary:logistic # 
  ################################################
  
  #In order to run the data through XGBoost, we need to first transform it to a dataframe
  Corp_Bundestag_V2_dfm_dataframe <- Corp_Bundestag_V2_dfm %>% 
    convert(to="data.frame")

  #Party Class Weights: how many speeches did each party give? Create a weight variable
  #additionally, we resample the training set and balance it with the SMOTE technique
  party_labels <- as.factor(Corp_Bundestag_V2$party)
  table(party_labels)
  weight_CDU        <- nrow(Corp_Bundestag_V2) / (5* nrow(subset(Corp_Bundestag_V2, Corp_Bundestag_V2$party == "CDU/CSU")))
  weight_FDP        <- nrow(Corp_Bundestag_V2) / (5* nrow(subset(Corp_Bundestag_V2, Corp_Bundestag_V2$party == "FDP")))
  weight_GRUENE     <- nrow(Corp_Bundestag_V2) / (5* nrow(subset(Corp_Bundestag_V2, Corp_Bundestag_V2$party == "GRUENE")))
  weight_SPD        <- nrow(Corp_Bundestag_V2) / (5* nrow(subset(Corp_Bundestag_V2, Corp_Bundestag_V2$party == "SPD")))
  weight_PDS_LINKE  <- nrow(Corp_Bundestag_V2) / (5* nrow(subset(Corp_Bundestag_V2, Corp_Bundestag_V2$party == "PDS/LINKE")))
  party_weights <- dplyr::recode(Corp_Bundestag_V2$party, "CDU/CSU" = weight_CDU, "FDP" = weight_FDP, 
                                 "GRUENE"=weight_GRUENE, "SPD"=weight_SPD,"PDS/LINKE"=weight_PDS_LINKE)

  #Rename and add party lables as outcome variables
  names(Corp_Bundestag_V2_dfm_dataframe)[1] <- "outcome_party"
  Corp_Bundestag_V2_dfm_dataframe$outcome_party <- Corp_Bundestag_V2$party
  names(Corp_Bundestag_V2_dfm_dataframe) <- make.names(names(Corp_Bundestag_V2_dfm_dataframe))
  
  #If you want to use Resampling methods, run the following code (this costs considerable computing resources): 
  #tryCatch( { Corp_Bundestag_V2_dfm_dataframe <- smote(outcome_party ~ ., Corp_Bundestag_V2_dfm_dataframe, perc.over = 10, perc.under=2) }
  #          , error = function(e) {usethis::edit_r_environ()})
  #if it runs an error, add: R_MAX_VSIZE=100Gb to your Renviron 
  #Corp_Bundestag_V2_dfm_dataframe <- smote(outcome_party ~ ., Corp_Bundestag_V2_dfm_dataframe, perc.over = 10, perc.under=2)
  
  #Save tokens as feature names
  feature_names <- as.data.frame(colnames(Corp_Bundestag_V2_dfm_dataframe[,-1]))
  colnames(feature_names) <- "features"
  feature_names <- feature_names$features
  
  #Create a numeric object that can be used as input for XGBoost
  complete.dfm.boost <- sapply(Corp_Bundestag_V2_dfm_dataframe[,-1], as.numeric)
  complete_labels.boost_multinomial <- as.numeric(as.factor(Corp_Bundestag_V2_dfm_dataframe[,1]))-1
  
  ## Cross-Validation XGBOOST
  boost.cv <-xgb.cv(data = xgb.DMatrix(complete.dfm.boost[,-1], label=complete_labels.boost_multinomial),
                                  max.depth=6,
                                  eta = 0.3,
                                  nround=100, 
                                  nfold=5,
                                  objective='multi:softmax',
                                  verbose=1,num_class=5, 
                                  eval_metric = "merror", 
                                  prediction = TRUE,
                                  stratified = TRUE,
                                  scale_pos_weight = party_weights)
  library(dplyr)
  #Save Predictions -> For each Party one Variable
  boost.pred.cv <- boost.cv$pred
  boost.pred.cv <- boost.pred.cv[,1]
  boost.pred.cv <- boost.pred.cv %>% 
    dplyr::recode("0" = "CDU/CSU",
                  "1" = "FDP",
                  "2" = "GRUENE",
                  "3" = "PDS/LINKE",
                  "4" = "SPD")
  table(boost.pred.cv, Corp_Bundestag_V2$party)
  prop.table(table(boost.pred.cv==Corp_Bundestag_V2$party))
  chisq.test(boost.pred.cv, Corp_Bundestag_V2$party)

  ####################
  # Confusion Matrix # -> for overall accuracy
  ####################
  #Micro-F1 -> overall accuracy
  #Macro-F1 -> F1-score per class averaged (unweighted) -> useful when there are imbalanced classes, which is the case here
  
  confusionMatrix(reference = as.factor(Corp_Bundestag_V2$party), data = as.factor(boost.pred.cv), mode='everything', positive='MM')
  
  ############################
  # Calculate Macro F1-Score # -> to calculate macro F1-Score, macro recall score
  ############################
  cm <- as.matrix(confusionMatrix(as.factor(boost.pred.cv), as.factor(Corp_Bundestag_V2$party)))
  n = sum(cm) # number of instances
  nc = nrow(cm) # number of classes
  rowsums = apply(cm, 1, sum) # number of instances per class
  colsums = apply(cm, 2, sum) # number of predictions per class
  diag = diag(cm)  # number of correctly classified instances per class 
  p = rowsums / n # distribution of instances over the actual classes
  q = colsums / n # distribution of instances over the predicted classes
  accuracy = sum(diag) / n 
  oneVsAll = lapply(1 : nc,
                    function(i){
                      v = c(cm[i,i],
                            rowsums[i] - cm[i,i],
                            colsums[i] - cm[i,i],
                            n-rowsums[i] - colsums[i] + cm[i,i]);
                      return(matrix(v, nrow = 2, byrow = T))})
  s = matrix(0, nrow = 2, ncol = 2)
  for(i in 1 : nc){s = s + oneVsAll[[i]]}
  micro_prf = (diag(s) / apply(s,1, sum))[1]
  micro_prf
  
  precision = diag / colsums 
  recall = diag / rowsums 
  f1 = 2 * precision * recall / (precision + recall) 
  
  print(" ************ Confusion Matrix ************")
  print(cm)
  print(" ************ Diag ************")
  print(diag)
  print(" ************ Precision/Recall/F1 ************")
  print(data.frame(precision, recall, f1)) 
  #Calculate Macro f1-score
  mean(recall)
  
  
  ##########
  # Graphs # 
  ##########
  # Overall party predictions over time 
  multi <- as.data.frame(cbind(boost.pred.cv, Corp_Bundestag_V2$party))
  colnames(multi) <- c('prediction','realout')
  multi$time <- as.Date(Corp_Bundestag_V2$date)

  # Create a month and year variable for averaging polls by approximate date
  multi <- multi %>%
    mutate(date = ymd(time),
           month = month(date),
           yr = year(date))
  
  # Now group the polls by their month and year, then summarise
  multi$accuracy_green <- if_else(multi$prediction == multi$realout & multi$realout == "GRUENE", 1,0)
  multi$accuracy_SPD <- if_else(multi$prediction == multi$realout & multi$realout == "SPD", 1,0)
  multi$accuracy_CDU <- if_else(multi$prediction == multi$realout & multi$realout == "CDU/CSU", 1,0)
  multi$accuracy_FDP <- if_else(multi$prediction == multi$realout & multi$realout == "FDP", 1,0)
  multi$accuracy_LINKE <- if_else(multi$prediction == multi$realout & multi$realout == "PDS/LINKE", 1,0)
  multi$accuracy_overall <- if_else(multi$prediction == multi$realout, 1,0)
  
  #Average for every party per month 
  #Green
  multi_green <- subset(multi, realout == "GRUENE")
  
  monthly_average_pred_green <- multi_green %>%
    group_by(yr, month) %>%
    summarise(avgaccuracy_green = mean(accuracy_green))
  
  #SPD
  multi_SPD <- subset(multi, realout == "SPD")
  
  monthly_average_pred_SPD <- multi_SPD %>%
    group_by(yr, month) %>%
    summarise(avgaccuracy_SPD = mean(accuracy_SPD))
  
  #CDU
  multi_CDU <- subset(multi, realout == "CDU/CSU")
  
  monthly_average_pred_CDU <- multi_CDU %>%
    group_by(yr, month) %>%
    summarise(avgaccuracy_CDU = mean(accuracy_CDU))
  
  #FDP
  multi_FDP <- subset(multi, realout == "FDP")
  
  monthly_average_pred_FDP <- multi_FDP %>%
    group_by(yr, month) %>%
    summarise(avgaccuracy_FDP = mean(accuracy_FDP))
  
  #DIE LINKE
  multi_LINKE <- subset(multi, realout == "PDS/LINKE")
  
  monthly_average_pred_LINKE <- multi_LINKE %>%
    group_by(yr, month) %>%
    summarise(avgaccuracy_LINKE = mean(accuracy_LINKE))
  
  ####################
  # COMBINE MEASURES #
  ####################
  monthly_average_pred_green <- left_join(monthly_average_pred_green, monthly_average_pred_SPD, by = c('yr','month'))
  monthly_average_pred_green <- left_join(monthly_average_pred_green, monthly_average_pred_CDU, by = c('yr','month'))
  monthly_average_pred_green <- left_join(monthly_average_pred_green, monthly_average_pred_FDP, by = c('yr','month'))
  monthly_average_pred_green <- left_join(monthly_average_pred_green, monthly_average_pred_LINKE, by = c('yr','month'))
  
  colnames(monthly_average_pred_green)

  ########
  # MELT #
  ########
  colnames(monthly_average_pred_green)
  monthly_average_pred_green <- monthly_average_pred_green %>%
    mutate(time = sprintf("%s-%s-%s",yr,month,"01"))
  
  monthly_average_pred_green.long <- melt(monthly_average_pred_green, id = c("time"), measure = c("avgaccuracy_green", "avgaccuracy_CDU","avgaccuracy_SPD", "avgaccuracy_FDP", "avgaccuracy_LINKE"))
  monthly_average_pred_green.long$time <- as.Date(monthly_average_pred_green.long$time)
  colnames(monthly_average_pred_green.long) <- c('time', 'Party','value')
  
  #Recode into different parties 
  monthly_average_pred_green.long$Party <- dplyr::recode(monthly_average_pred_green.long$Party , 'avgaccuracy_CDU' = 'CDU/CSU', 'avgaccuracy_FDP' = 'FDP', 'avgaccuracy_green'="GRUENE","avgaccuracy_SPD"="SPD","avgaccuracy_LINKE"="LINKE")
  
  ###########
  # ggplot2 #
  ###########
  g1 <- ggplot(data = monthly_average_pred_green.long, aes(x=time,y=value, color = Party))+ geom_point(alpha = 0.2) +
    geom_smooth(span = 0.2, alpha = 0.1, aes(fill = Party)) +
    scale_x_date(date_breaks = "6 months",
                 date_minor_breaks = "1 month", date_labels = "%Y-%m") + theme_bw() + xlab("Year") + ylab("Monthly Average of True Positive Rates") + scale_fill_manual(values = c("green3","black","red3", '#F0E442','purple')) +
    scale_color_manual(values = c("green3","black","red3", '#F0E442','purple'))

  g1
  
  
  ####################
  # Rolling Averages #
  ####################
  
  multi_green <- multi_green %>%
    mutate(roll_avgaccuracy_green = rollapply(accuracy_green, 100, FUN = mean, fill = NA, align = "left", partial = TRUE)) %>% group_by(date) %>%
    summarise(roll_avgaccuracy_green = mean(roll_avgaccuracy_green))
  multi_SPD <- multi_SPD %>%
    mutate(roll_avgaccuracy_SPD = rollapply(accuracy_SPD, 100, FUN = mean, fill = NA, align = "left", partial = TRUE)) %>% group_by(date) %>%
    summarise(roll_avgaccuracy_SPD = mean(roll_avgaccuracy_SPD))
  multi_CDU <- multi_CDU %>%
    mutate(roll_accuracy_CDU = rollapply(accuracy_CDU, 100, FUN = mean, fill = NA, align = "left", partial = TRUE)) %>% group_by(date) %>%
    summarise(roll_accuracy_CDU = mean(roll_accuracy_CDU))
  multi_FDP <- multi_FDP %>%
    mutate(roll_accuracy_FDP = rollapply(accuracy_FDP, 100, FUN = mean, fill = NA, align = "left", partial = TRUE)) %>% group_by(date) %>%
    summarise(roll_accuracy_FDP = mean(roll_accuracy_FDP))
  multi_LINKE <- multi_LINKE %>%
    mutate(roll_avgaccuracy_LINKE = rollapply(accuracy_LINKE, 100, FUN = mean, fill = NA, align = "left", partial = TRUE)) %>% group_by(date) %>%
    summarise(roll_avgaccuracy_LINKE = mean(roll_avgaccuracy_LINKE))
  
  ####################
  # COMBINE MEASURES #
  ####################
  speech_dates <- as.data.frame(unique(multi$date))
  colnames(speech_dates) <- 'date'
  multi_rollavg <- left_join(speech_dates, multi_green, by = c('date'))
  multi_rollavg <- left_join(multi_rollavg, multi_SPD, by = c('date'))
  multi_rollavg <- left_join(multi_rollavg, multi_CDU, by = c('date'))
  multi_rollavg <- left_join(multi_rollavg, multi_FDP, by = c('date'))
  multi_rollavg <- left_join(multi_rollavg, multi_LINKE, by = c('date'))
  
  multi_rollavg.long <- melt(multi_rollavg, id = c("date"), measure = c("roll_avgaccuracy_green", "roll_accuracy_CDU","roll_avgaccuracy_SPD", "roll_accuracy_FDP", "roll_avgaccuracy_LINKE"))
  multi_rollavg.long$time <- as.Date(multi_rollavg.long$date)
  colnames(multi_rollavg.long) <- c('time', 'Party','value')
  
  #Recode into different parties 
  multi_rollavg.long$Party <- dplyr::recode(multi_rollavg.long$Party , 'roll_accuracy_CDU' = 'CDU/CSU', 'roll_accuracy_FDP' = 'FDP', 'roll_avgaccuracy_green'="GRUENE","roll_avgaccuracy_SPD"="SPD","roll_avgaccuracy_LINKE"="LINKE")
  
  ###########
  # ggplot2 #
  ###########
  g2 <- ggplot(data = multi_rollavg.long, aes(x=time,y=value, color = Party))+ geom_point(alpha = 0.2) +
    geom_smooth(span = 0.2, alpha = 0.1, aes(fill = Party)) +
    scale_x_date(date_breaks = "6 months",
                 date_minor_breaks = "1 month", date_labels = "%Y-%m") + theme_bw() + xlab("Year") + ylab("Rolling Average of True Positive Rates") + scale_fill_manual(values = c("green3","black","red3", '#F0E442','purple')) +
    scale_color_manual(values = c("green3","black","red3", '#F0E442','purple'))
  
  g2
  

  ####
  # If you would like to run One vs. the Rest (OVR), you can also calculate several binomial models and use their predictions separately:
  ##
  ##Cross-Validation XGBOOST
  #Create several binary outcome variables (faster, better training than multinomial)
  complete_labels.boost_multinomial_CDU_CSU <- dplyr::recode(complete_labels.boost_multinomial, "0" = "1",
                                                             "1" = "0",
                                                             "2" = "0",
                                                             "3" = "0",
                                                             "4" = "0")
  complete_labels.boost_multinomial_SPD <- dplyr::recode(complete_labels.boost_multinomial, "0" = "0",
                                                        "1" = "0",
                                                        "2" = "0",
                                                        "3" = "0",
                                                        "4" = "1")
  complete_labels.boost_multinomial_GRUENE <- dplyr::recode(complete_labels.boost_multinomial, "0" = "0",
                                                            "1" = "0",
                                                            "2" = "1",
                                                            "3" = "0",
                                                            "4" = "0")
  complete_labels.boost_multinomial_PDS_LINKE <- dplyr::recode(complete_labels.boost_multinomial, "0" = "0",
                                                               "1" = "0",
                                                               "2" = "0",
                                                               "3" = "1",
                                                               "4" = "0")
  complete_labels.boost_multinomial_FDP <- dplyr::recode(complete_labels.boost_multinomial, "0" = "0",
                                                         "1" = "1",
                                                         "2" = "0",
                                                         "3" = "0",
                                                         "4" = "0")
  
  library(xgboost)

  #Use this if you want to save the training data
  #dtrain <- xgb.DMatrix(complete.dfm.boost[,-1], label=complete_labels.boost_multinomial_ADDPARTYHERE)
  
  #xgb.DMatrix.save(dtrain, 'xgb.DMatrix.data')
  
  #Use this if you want to load the saved training data
  #dtrain <- xgb.DMatrix('xgb.DMatrix.data')
  
  #One could perform hyperparameter tuning here with grid.search (see code further down below)
  #Models take 1-2 hours to train, adjust the hyperparameters if you want to train faster
  boost.cv_CDU_CSU <-xgb.cv(data = xgb.DMatrix(complete.dfm.boost[,-1], label=complete_labels.boost_multinomial_CDU_CSU),
                                  max.depth=1, #depth of the trees at 1 (lower for faster training)
                                  eta = 0.2, #learning rate at 0.2 (set higher for faster training)
                                  nround= 100, #training for 100 rounds (set lower for faster training)
                                  nfold=5, #five fold cross validation
                                  objective='binary:logistic',
                                  verbose=1, #see model training specifics
                                  prediction = TRUE, #save predictions from the model
                                  stratified = TRUE, #stratify the sample
                    ) 
  boost.cv_SPD <-xgb.cv(data = xgb.DMatrix(complete.dfm.boost[,-1], label=complete_labels.boost_multinomial_SPD),
                            max.depth=1, #depth of the trees at 1 (lower for faster training)
                            eta = 0.2, #learning rate at 0.2 (set higher for faster training)
                            nround= 100, #training for 100 rounds (set lower for faster training)
                            nfold=5, #five fold cross validation
                            objective='binary:logistic',
                            verbose=1, #see model training specifics
                            prediction = TRUE, #save predictions from the model
                            stratified = TRUE, #stratify the sample
  ) 
  boost.cv_GRUENE <-xgb.cv(data = xgb.DMatrix(complete.dfm.boost[,-1], label=complete_labels.boost_multinomial_GRUENE),
                            max.depth=1, #depth of the trees at 1 (lower for faster training)
                            eta = 0.2, #learning rate at 0.2 (set higher for faster training)
                            nround= 100, #training for 100 rounds (set lower for faster training)
                            nfold=5, #five fold cross validation
                            objective='binary:logistic',
                            verbose=1, #see model training specifics
                            prediction = TRUE, #save predictions from the model
                            stratified = TRUE, #stratify the sample
  ) 
  boost.cv_PDS_LINKE <-xgb.cv(data = xgb.DMatrix(complete.dfm.boost[,-1], label=complete_labels.boost_multinomial_PDS_LINKE),
                            max.depth=1, #depth of the trees at 1 (lower for faster training)
                            eta = 0.2, #learning rate at 0.2 (set higher for faster training)
                            nround= 100, #training for 100 rounds (set lower for faster training)
                            nfold=5, #five fold cross validation
                            objective='binary:logistic',
                            verbose=1, #see model training specifics
                            prediction = TRUE, #save predictions from the model
                            stratified = TRUE, #stratify the sample
  ) 
  boost.cv_FDP <-xgb.cv(data = xgb.DMatrix(complete.dfm.boost[,-1], label=complete_labels.boost_multinomial_FDP),
                            max.depth=1, #depth of the trees at 1 (lower for faster training)
                            eta = 0.2, #learning rate at 0.2 (set higher for faster training)
                            nround= 100, #training for 100 rounds (set lower for faster training)
                            nfold=5, #five fold cross validation
                            objective='binary:logistic',
                            verbose=1, #see model training specifics
                            prediction = TRUE, #save predictions from the model
                            stratified = TRUE, #stratify the sample
  ) 

  boost.pred.cv_CDU_CSU <- boost.cv_CDU_CSU$pred
  
  boost.pred.cv_SPD <- boost.cv_SPD$pred

  boost.pred.cv_GRUENE <- boost.cv_GRUENE$pred

  boost.pred.cv_PDS_LINKE <- boost.cv_PDS_LINKE$pred

  boost.pred.cv_FDP <- boost.cv_FDP$pred

  
  ##########
  # Graphs # 
  ##########
  # Overall party predictions over time 
  multi_binomial <- as.data.frame(cbind(boost.pred.cv_CDU_CSU, boost.pred.cv_SPD, boost.pred.cv_GRUENE, boost.pred.cv_PDS_LINKE, boost.pred.cv_FDP, Corp_Bundestag_V2$party))
  colnames(multi_binomial) <- c('accuracy_CDU', 'accuracy_SPD', 'accuracy_green', 'accuracy_LINKE', 'accuracy_FDP', 'realout')
  multi_binomial$time <- as.Date(Corp_Bundestag_V2$date)
  
  # Create a month and year variable for averaging polls by approximate date
  multi_binomial <- multi_binomial %>%
    mutate(date = ymd(time),
           month = month(date),
           yr = year(date))

  #Average for every party per month 
  #Green
  multi_binomial_green <- subset(multi_binomial, realout == "GRUENE")
  
  monthly_average_binomial_pred_green <- multi_binomial_green %>%
    group_by(yr, month) %>%
    summarise(avgaccuracy_green = mean(as.numeric(accuracy_green)))
  
  #SPD
  multi_binomial_SPD <- subset(multi_binomial, realout == "SPD")
  
  monthly_average_binomial_pred_SPD <- multi_binomial_SPD %>%
    group_by(yr, month) %>%
    summarise(avgaccuracy_SPD = mean(as.numeric(accuracy_SPD)))
  
  #CDU
  multi_binomial_CDU <- subset(multi_binomial, realout == "CDU/CSU")
  
  monthly_average_binomial_pred_CDU <- multi_binomial_CDU %>%
    group_by(yr, month) %>%
    summarise(avgaccuracy_CDU = mean(as.numeric(accuracy_CDU)))
  
  #FDP
  multi_binomial_FDP <- subset(multi_binomial, realout == "FDP")
  
  monthly_average_binomial_pred_FDP <- multi_binomial_FDP %>%
    group_by(yr, month) %>%
    summarise(avgaccuracy_FDP = mean(as.numeric(accuracy_FDP)))
  
  #DIE LINKE
  multi_binomial_LINKE <- subset(multi_binomial, realout == "PDS/LINKE")
  
  monthly_average_binomial_pred_LINKE <- multi_binomial_LINKE %>%
    group_by(yr, month) %>%
    summarise(avgaccuracy_LINKE = mean(as.numeric(accuracy_LINKE)))
  
  ####################
  # COMBINE MEASURES #
  ####################
  monthly_average_binomial_pred_green <- left_join(monthly_average_binomial_pred_green, monthly_average_binomial_pred_SPD, by = c('yr','month'))
  monthly_average_binomial_pred_green <- left_join(monthly_average_binomial_pred_green, monthly_average_binomial_pred_CDU, by = c('yr','month'))
  monthly_average_binomial_pred_green <- left_join(monthly_average_binomial_pred_green, monthly_average_binomial_pred_FDP, by = c('yr','month'))
  monthly_average_binomial_pred_green <- left_join(monthly_average_binomial_pred_green, monthly_average_binomial_pred_LINKE, by = c('yr','month'))
  
  colnames(monthly_average_binomial_pred_green)
  
  ########
  # MELT #
  ########
  colnames(monthly_average_binomial_pred_green)
  monthly_average_binomial_pred_green <- monthly_average_binomial_pred_green %>%
    mutate(time = sprintf("%s-%s-%s",yr,month,"01"))
  
  monthly_average_binomial_pred_green.long <- melt(monthly_average_binomial_pred_green, id = c("time"), measure = c("avgaccuracy_green", "avgaccuracy_CDU","avgaccuracy_SPD", "avgaccuracy_FDP", "avgaccuracy_LINKE"))
  monthly_average_binomial_pred_green.long$time <- as.Date(monthly_average_binomial_pred_green.long$time)
  colnames(monthly_average_binomial_pred_green.long) <- c('time', 'Party','value')
  
  #Recode into different parties 
  monthly_average_binomial_pred_green.long$Party <- dplyr::recode(monthly_average_binomial_pred_green.long$Party , 'avgaccuracy_CDU' = 'CDU/CSU', 'avgaccuracy_FDP' = 'FDP', 'avgaccuracy_green'="GRUENE","avgaccuracy_SPD"="SPD","avgaccuracy_LINKE"="LINKE")
  
  ###########
  # ggplot2 #
  ###########
  g3 <- ggplot(data = monthly_average_binomial_pred_green.long, aes(x=time,y=value, color = Party))+ geom_point(alpha = 0.2) +
    geom_smooth(span = 0.2, alpha = 0.1, aes(fill = Party)) +
    scale_x_date(date_breaks = "6 months",
                 date_minor_breaks = "1 month", date_labels = "%Y-%m") + theme_bw() + xlab("Year") + ylab("Monthly Average of True Positive Rates") + scale_fill_manual(values = c("green3","black","red3", '#F0E442','purple')) +
    scale_color_manual(values = c("green3","black","red3", '#F0E442','purple'))
  
  g3
  
  
  ####################
  # Rolling Averages #
  ####################
  
  multi_binomial_green <- multi_binomial_green %>%
    mutate(roll_avgaccuracy_green = rollapply(as.numeric(accuracy_green), 100, FUN = mean, fill = NA, align = "left", partial = TRUE)) %>% group_by(date) %>%
    summarise(roll_avgaccuracy_green = mean(roll_avgaccuracy_green))
  multi_binomial_SPD <- multi_binomial_SPD %>%
    mutate(roll_avgaccuracy_SPD = rollapply(as.numeric(accuracy_SPD), 100, FUN = mean, fill = NA, align = "left", partial = TRUE)) %>% group_by(date) %>%
    summarise(roll_avgaccuracy_SPD = mean(roll_avgaccuracy_SPD))
  multi_binomial_CDU <- multi_binomial_CDU %>%
    mutate(roll_accuracy_CDU = rollapply(as.numeric(accuracy_CDU), 100, FUN = mean, fill = NA, align = "left", partial = TRUE)) %>% group_by(date) %>%
    summarise(roll_accuracy_CDU = mean(roll_accuracy_CDU))
  multi_binomial_FDP <- multi_binomial_FDP %>%
    mutate(roll_accuracy_FDP = rollapply(as.numeric(accuracy_FDP), 100, FUN = mean, fill = NA, align = "left", partial = TRUE)) %>% group_by(date) %>%
    summarise(roll_accuracy_FDP = mean(roll_accuracy_FDP))
  multi_binomial_LINKE <- multi_binomial_LINKE %>%
    mutate(roll_avgaccuracy_LINKE = rollapply(as.numeric(accuracy_LINKE), 100, FUN = mean, fill = NA, align = "left", partial = TRUE)) %>% group_by(date) %>%
    summarise(roll_avgaccuracy_LINKE = mean(roll_avgaccuracy_LINKE))
  
  ####################
  # COMBINE MEASURES #
  ####################
  speech_dates_binomial <- as.data.frame(unique(multi_binomial$date))
  colnames(speech_dates_binomial) <- 'date'
  multi_binomial_rollavg <- left_join(speech_dates_binomial, multi_binomial_green, by = c('date'))
  multi_binomial_rollavg <- left_join(multi_binomial_rollavg, multi_binomial_SPD, by = c('date'))
  multi_binomial_rollavg <- left_join(multi_binomial_rollavg, multi_binomial_CDU, by = c('date'))
  multi_binomial_rollavg <- left_join(multi_binomial_rollavg, multi_binomial_FDP, by = c('date'))
  multi_binomial_rollavg <- left_join(multi_binomial_rollavg, multi_binomial_LINKE, by = c('date'))
  
  multi_binomial_rollavg.long <- melt(multi_binomial_rollavg, id = c("date"), measure = c("roll_avgaccuracy_green", "roll_accuracy_CDU","roll_avgaccuracy_SPD", "roll_accuracy_FDP", "roll_avgaccuracy_LINKE"))
  multi_binomial_rollavg.long$time <- as.Date(multi_binomial_rollavg.long$date)
  colnames(multi_binomial_rollavg.long) <- c('time', 'Party','value')
  
  #Recode into different parties 
  multi_binomial_rollavg.long$Party <- dplyr::recode(multi_binomial_rollavg.long$Party , 'roll_accuracy_CDU' = 'CDU/CSU', 'roll_accuracy_FDP' = 'FDP', 'roll_avgaccuracy_green'="GRUENE","roll_avgaccuracy_SPD"="SPD","roll_avgaccuracy_LINKE"="LINKE")
  
  ###########
  # ggplot2 #
  ###########
  g4 <- ggplot(data = multi_binomial_rollavg.long, aes(x=time,y=value, color = Party))+ geom_point(alpha = 0.2) +
    geom_smooth(span = 0.2, alpha = 0.1, aes(fill = Party)) +
    scale_x_date(date_breaks = "6 months",
                 date_minor_breaks = "1 month", date_labels = "%Y-%m") + theme_bw() + xlab("Year") + ylab("Rolling Average of True Positive Rates") + scale_fill_manual(values = c("green3","black","red3", '#F0E442','purple')) +
    scale_color_manual(values = c("green3","black","red3", '#F0E442','purple'))
  
  g4
  