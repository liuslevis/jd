library(randomForest)

action <- read.csv('data/raw/JData_Action_201603.csv', colClasses=c('numeric', 'numeric', 'character', 'factor', 'factor', 'factor', 'factor'))


combi <- read.csv('data/input/v1/train_20160201_20160229_20160301_20160305.csv', colClasses=c('factor','numeric','numeric',rep('numeric', 6), rep('factor',30)))
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

> summary(combi) #new
> summary(combi)
 label        user_id           sku_id           act_1
 0:51129   Min.   :200002   Min.   :    95   Min.   :  0.000
 1:  599   1st Qu.:227751   1st Qu.: 39253   1st Qu.:  2.000
           Median :253112   Median : 78694   Median :  2.000
           Mean   :253338   Mean   : 83042   Mean   :  5.413
           3rd Qu.:279202   3rd Qu.:128988   3rd Qu.:  6.000
           Max.   :305321   Max.   :171182   Max.   :408.000
     act_2             act_3              act_4             act_5
 Min.   : 0.0000   Min.   : 0.00000   Min.   :0.00000   Min.   :0.00000
 1st Qu.: 0.0000   1st Qu.: 0.00000   1st Qu.:0.00000   1st Qu.:0.00000
 Median : 0.0000   Median : 0.00000   Median :0.00000   Median :0.00000
 Mean   : 0.2406   Mean   : 0.06018   Mean   :0.02987   Mean   :0.04578
 3rd Qu.: 0.0000   3rd Qu.: 0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000
 Max.   :32.0000   Max.   :27.00000   Max.   :3.00000   Max.   :5.00000
     act_6         user_sex_.1 user_sex_0 user_sex_1 user_sex_2 user_age_.1
 Min.   :  0.000   0:51727     0:29433    0:47851    0:26173    0:45021
 1st Qu.:  2.000   1:    1     1:22295    1: 3877    1:25555    1: 6707
 Median :  4.000
 Mean   :  7.811
 3rd Qu.:  8.000
 Max.   :411.000
 user_age_1 user_age_2 user_age_3 user_age_4 user_age_5 user_age_6 user_lv_cd_1
 0:51724    0:47396    0:28166    0:37120    0:50130    0:50811    0:50674
 1:    4    1: 4332    1:23562    1:14608    1: 1598    1:  917    1: 1054

 user_lv_cd_2 user_lv_cd_3 user_lv_cd_4 user_lv_cd_5 user_reg_tm_0
 0:47687      0:40604      0:35592      0:32355      0:51720
 1: 4041      1:11124      1:16136      1:19373      1:    8

 user_reg_tm_1 user_reg_tm_2 user_reg_tm_3 user_reg_tm_4 user_reg_tm_5
 0:50732       0:46376       0:46837       0:43128       0:19847
 1:  996       1: 5352       1: 4891       1: 8600       1:31881

 sku_a1_.1 sku_a1_1  sku_a1_2  sku_a1_3  sku_a2_.1 sku_a2_1  sku_a2_2
 0:50831   0:28612   0:49597   0:26144   0:49244   0:16696   0:37516
 1:  897   1:23116   1: 2131   1:25584   1: 2484   1:35032   1:14212

 sku_a3_.1 sku_a3_1  sku_a3_2  user_a1_.1 user_a1_1 user_a1_2 user_a1_3
 0:47259   0:30724   0:25473   0:50831    0:28612   0:49597   0:26144
 1: 4469   1:21004   1:26255   1:  897    1:23116   1: 2131   1:25584

 user_a2_.1 user_a2_1 user_a2_2 user_a3_.1 user_a3_1   user_a3_2
 0:49244    0:16696   0:37516   0:47259    0:30724   Min.   :0.0000
 1: 2484    1:35032   1:14212   1: 4469    1:21004   1st Qu.:0.0000
                                                     Median :1.0000
                                                     Mean   :0.5076
                                                     3rd Qu.:1.0000
                                                     Max.   :1.0000

> combi <- read.csv('data/input/v1/train_20160201_20160229_20160301_20160305.csv', colClasses=c('factor','numeric','numeric',rep('numeric', 6), rep('factor',30)))
> summary(combi)
 label        user_id           sku_id           act_1
 0:50730   Min.   :200002   Min.   :    95   Min.   :  0.000
 1:  998   1st Qu.:227751   1st Qu.: 39253   1st Qu.:  2.000
           Median :253112   Median : 78694   Median :  2.000
           Mean   :253338   Mean   : 83048   Mean   :  5.403
           3rd Qu.:279202   3rd Qu.:128988   3rd Qu.:  6.000
           Max.   :305321   Max.   :171182   Max.   :408.000
     act_2             act_3              act_4             act_5
 Min.   : 0.0000   Min.   : 0.00000   Min.   :0.00000   Min.   :0.00000
 1st Qu.: 0.0000   1st Qu.: 0.00000   1st Qu.:0.00000   1st Qu.:0.00000
 Median : 0.0000   Median : 0.00000   Median :0.00000   Median :0.00000
 Mean   : 0.2396   Mean   : 0.05979   Mean   :0.02987   Mean   :0.04599
 3rd Qu.: 0.0000   3rd Qu.: 0.00000   3rd Qu.:0.00000   3rd Qu.:0.00000
 Max.   :32.0000   Max.   :27.00000   Max.   :3.00000   Max.   :5.00000
     act_6         user_sex_.2147483648 user_sex_0 user_sex_1 user_sex_2
 Min.   :  0.000   0:51727              0:29433    0:47851    0:26173
 1st Qu.:  2.000   1:    1              1:22295    1: 3877    1:25555
 Median :  4.000
 Mean   :  7.801
 3rd Qu.:  8.000
 Max.   :411.000
 user_age_.1 user_age_0 user_age_1 user_age_2 user_age_3 user_age_4 user_age_5
 0:51727     0:45022    0:51724    0:47396    0:28166    0:37120    0:50130
 1:    1     1: 6706    1:    4    1: 4332    1:23562    1:14608    1: 1598




 user_age_6 user_lv_cd_1 user_lv_cd_2 user_lv_cd_3 user_lv_cd_4 user_lv_cd_5
 0:50811    0:50674      0:47687      0:40604      0:35592      0:32355
 1:  917    1: 1054      1: 4041      1:11124      1:16136      1:19373




 user_reg_tm_0 user_reg_tm_1 user_reg_tm_2 user_reg_tm_3 user_reg_tm_4
 0:51720       0:50732       0:46376       0:46837       0:43128
 1:    8       1:  996       1: 5352       1: 4891       1: 8600




 user_reg_tm_5 sku_a1_.1 sku_a1_1  sku_a1_2  sku_a1_3  sku_a2_.1 sku_a2_1
 0:19847       0:50832   0:28598   0:49603   0:26151   0:49238   0:16688
 1:31881       1:  896   1:23130   1: 2125   1:25577   1: 2490   1:35040




 sku_a2_2  sku_a3_.1    sku_a3_1         sku_a3_2
 0:37530   0:47269   Min.   :0.0000   Min.   :0.0000
 1:14198   1: 4459   1st Qu.:0.0000   1st Qu.:0.0000
                     Median :0.0000   Median :1.0000
                     Mean   :0.4061   Mean   :0.5077
                     3rd Qu.:1.0000   3rd Qu.:1.0000
                     Max.   :1.0000   Max.   :1.0000
>