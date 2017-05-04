library(randomForest)
combi <- read.csv('data/input/train_20160201_20160229_20160301_20160305.csv', colClasses=c('factor','numeric','numeric',rep('numeric', 6), rep('factor',40)))
summary(combi)

set.seed(123)
train_ind <- sample(seq_len(nrow(combi)), size = floor(0.75 * nrow(combi)))

train <- combi[train_ind, ]
test <- combi[-train_ind, ]
formula <- label ~  act_1 + act_2 + act_3 + act_4 + act_5 + act_6 + user_sex_.2147483648 + user_sex_0 + user_sex_1 + user_sex_2 + user_age_.1 + user_age_0 + user_age_1 + user_age_2 + user_age_3 + user_age_4 + user_age_5 + user_age_6 + user_lv_cd_1 + user_lv_cd_2 + user_lv_cd_3 + user_lv_cd_4 + user_lv_cd_5 + user_reg_tm_0 + user_reg_tm_1 + user_reg_tm_2 + user_reg_tm_3 + user_reg_tm_4 + user_reg_tm_5 + sku_a1_.1 + sku_a1_1 + sku_a1_2 + sku_a1_3 + sku_a2_.1 + sku_a2_1 + sku_a2_2 + sku_a3_.1 + sku_a3_1 + sku_a3_2 
rf <- randomForest(formula=formula, data=train, importance=TRUE, ntree=200, do.trace=0, nodesize=2**3)
rf
#Confusion matrix:
#       0   1  class.error
# 0 38028   3 7.888302e-05
# 1   355 410 4.640523e-01
varImpPlot(rf)

user <- read.csv('data/raw/JData_User.csv', colClasses=c('numeric', 'factor', 'factor', 'factor', 'factor'))
product <- read.csv('data/raw/JData_Product.csv', colClasses=c('numeric', 'factor', 'factor', 'factor', 'factor', 'factor'))
comment <- read.csv('data/raw/JData_Comment.csv', colClasses=c('factor', 'numeric', 'factor', 'factor', 'numeric'))
action <- read.csv('data/raw/JData_Action_201602.csv', colClasses=c('numeric', 'numeric', 'character', 'factor', 'factor', 'factor', 'factor'))

summary(user)
summary(product)
summary(comment)
summary(action)


user_id
    age
    sex
    user_lv_cd
    user_reg_tm

sku_id
    a1
    a2
    a3
    cate
    brand

comment
    dt
    sku_id
    comment_num
    has_bad_comment
    bad_comment_rate

action
    user_id
    sku_id
    time
    model_id
    type
    cate
    brand

> summary(user)
    user_id             age          sex        user_lv_cd     user_reg_tm
 Min.   :200001   26-35岁 :46570   0   :42846   1: 2666    2015-11-11:   412
 1st Qu.:226331   36-45岁 :30336   1   : 7737   2: 9661    2014-11-11:   348
 Median :252661   -1      :14412   2   :54735   3:24563    2014-06-05:   252
 Mean   :252661   16-25岁 : 8797   NULL:    3   4:32343    2015-06-18:   226
 3rd Qu.:278991   46-55岁 : 3325                5:36088    2013-06-18:   214
 Max.   :305321   56岁以上: 1871                           2014-06-06:   209
                  (Other) :   10                           (Other)   :103660

> summary(product)
     sku_id        a1         a2         a3        cate          brand
 Min.   :     6   -1: 1701   -1: 4050   -1: 3815   8:24187   489    :6637
 1st Qu.: 42476   1 : 4760   1 :13513   1 : 8394             214    :6444
 Median : 85616   2 : 3582   2 : 6624   2 :11978             623    :1101
 Mean   : 85399   3 :14144                                   812    :1061
 3rd Qu.:127774                                              800    :1015
 Max.   :171224                                              545    : 945
                                                             (Other):6984

> summary(comment)
          dt             sku_id       comment_num has_bad_comment
 2016-02-01: 46546   Min.   :     8   0: 19993    0:292978
 2016-02-08: 46546   1st Qu.: 43224   1: 85430    1:265574
 2016-02-15: 46546   Median : 85892   2:168698
 2016-02-22: 46546   Mean   : 85830   3:119642
 2016-02-29: 46546   3rd Qu.:128624   4:164789
 2016-03-07: 46546   Max.   :171225
 (Other)   :279276
 bad_comment_rate
 Min.   :0.00000
 1st Qu.:0.00000
 Median :0.00000
 Mean   :0.04999
 3rd Qu.:0.04650
 Max.   :1.00000

 > summary(action)
    user_id           sku_id           time              model_id
 Min.   :200002   Min.   :     2   Length:11485424           :4959617
 1st Qu.:227009   1st Qu.: 40845   Class :character   0      :2390069
 Median :251208   Median : 81163   Mode  :character   216    : 892298
 Mean   :252442   Mean   : 83791                      217    : 829246
 3rd Qu.:278288   3rd Qu.:127152                      218    : 587600
 Max.   :305321   Max.   :171225                      27     : 335810
                                                      (Other):1490784
 type             cate             brand
 1:4554840   8      :3194275   306    :1237436
 2: 140727   4      :2918671   489    :1116745
 3:  58917   6      :1642371   214    : 982388
 4:  11485   5      :1483361   800    : 568360
 5:  29030   7      :1126849   885    : 490275
 6:6690425   9      : 888435   545    : 468920
             (Other): 231462   (Other):6621300