# JDATA算法大赛入门(score0.07+时间滑动窗口特征＋xgboost模型)

## 依赖库

- pandas
- sklearn
- xgboost

## 项目结构

- data: 储存数据目录
- cache: 缓存目录
- sub: 结果目录
- train: 训练代码
- gen_feat: 生成特征

## 使用说明

转码

```
for FILE in 'JData_User.csv' 'JData_Product.csv' 'JData_Comment.csv' 'JData_Action_201602.csv' 'JData_Action_201603.csv' 'JData_Action_201604.csv'; do
    mv data/raw/${FILE} data/raw/${FILE}.gbk
    iconv -f gbk -t utf-8  data/raw/${FILE}.gbk > data/raw/${FILE}
done

trash data/raw/*.gbk
```

安装 xgboost

```
http://xgboost.readthedocs.io/en/latest/build.html#building-on-osx
```

生成特征，训练模型

```
python3 gen_feat.py 20160201 20160206 20160211 20160216 20160221 20160226 20160301 20160306 20160311 20160316 20160318
python3 train.py
```

## 特征工程

```
 user_id     用户ID    脱敏
    age_i     年龄段     -1表示未知
    sex_i     性别  0表示男，1表示女，2表示保密
    user_lv_cd_i  用户等级    有顺序的级别枚举，越高级别数字越大
    user_reg_tm_i     用户注册日期  粒度到天

sku_id  商品编号    脱敏
    a1_j  属性1     枚举，-1表示未知
    a2_j  属性2     枚举，-1表示未知
    a3_j  属性3     枚举，-1表示未知
    cate_j    品类ID    脱敏
    brand_j   品牌ID    脱敏
    comment_num_j
    has_bad_comment
    bad_comment_rate (dt)

train_data(d1,d2,d3,d4)
    user_id
    sku_id 
    label(d3,d4) 是否在未来购买 action type=4
    feat(d1,d2)
        # 用户画像属性
        user_feat 

        # 商品画像属性 
        sku_feat

        # 用户行为属性
        user_act_i

        # 商品行为属性
        sku_act_i

        # 用户对sku偏好
        user_sku_act_i

        # 用户对 a1 偏好
        user_a1_j_act_i

        # 用户对 a2 偏好
        user_a2_j_act_i
        
        # 用户对 a3 偏好
        user_a3_j_act_i

        # 用户对 cat 偏好
        user_cat_j_act_i

        # 用户对 brand 的偏好
        user_brand_j

        # 用户在月初、月中、月末的购买次数
        user_buy_month_stage_0~2
        # 在预测日期，属于月初、月中、还是月末？
        user_pred_month_stage_0~2

 ```
