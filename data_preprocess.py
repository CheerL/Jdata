import numpy as np
import pandas as pd

######################### 用户数据 #########################

user_file = 'data/JData_User.csv'
user_data = pd.read_csv(
    user_file,
    index_col='user_id',
    encoding='gbk'
)                                   # 105321, 3个用户的数据有缺失
user_data.dropna(inplace=True)      # 105318

##### 官方说明 #####
#  user_id          用户ID            脱敏
#  age              年龄段            -1表示未知
#  sex              性别              0表示男，1表示女，2表示保密
#  user_lv_cd       用户等级          有顺序的级别枚举，越高级别数字越大
#  user_reg_tm      用户注册日期       粒度到天

# age           = [-1, 26-35岁, 36-45岁, 16-25岁, 46-55岁, 56岁以上, 15岁以下]
# sex           = [0, 1, 2]
# user_lv_cd    = [1, 2, 3, 4, 5]

# 修改age为数值, 取年龄下限 [-1, 0, 16, 26, 36, 46, 56]
age_list_before = ['-1', '15岁以下', '16-25岁',
                   '26-35岁', '36-45岁', '46-55岁', '56岁以上']
age_list_after = [-1, 0, 16, 26, 36, 46, 56]
user_data['age'].replace(age_list_before, age_list_after, inplace=True)

user_no_age = user_data[user_data.age == -1]         # len = 14412
user_no_sex = user_data[user_data.sex == 2]          # len = 54735
user_no_sex_and_age = user_data[
    (user_data.sex == 2) & (user_data.age == -1)]    # len = 10299
user_full_info = user_data[
    (user_data.sex != 2) & (user_data.age != -1)]    # len = 46470

######################### 商品数据 #########################

product_file = 'data/JData_Product.csv'
product_data = pd.read_csv(
    product_file,
    index_col='sku_id'
    # encoding='gbk'
)                                   # 24187, 没有缺失值

##### 官方说明 #####
#  sku_id       商品编号               脱敏
#  a1           属性1                 枚举，-1表示未知
#  a2           属性2                 枚举，-1表示未知
#  a3           属性3                 枚举，-1表示未知
#  cate         品类ID                脱敏
#  brand        品牌ID                脱敏

##### 属性取值及注释 #####
# a1        = [-1, 1, 2, 3]
# a2        = [-1, 1, 2]
# a3        = [-1, 1, 2]
# cate      = [8],              只有一个值, 可以忽略掉了, 开心
# brand     = [...]             详见 product_data['brand'].value_counts()

pro_no_a1 = product_data[product_data.a1 == -1]       # len = 1701
pro_no_a2 = product_data[product_data.a2 == -1]       # len = 4050
pro_no_a3 = product_data[product_data.a3 == -1]       # len = 3815
pro_no_a1_a2 = product_data[
    (product_data.a1 == -1) &
    (product_data.a2 == -1)
]                                                      # len = 671
pro_no_a1_a3 = product_data[
    (product_data.a1 == -1) &
    (product_data.a3 == -1)
]                                                      # len = 375
pro_no_a2_a3 = product_data[
    (product_data.a2 == -1) &
    (product_data.a3 == -1)
]                                                      # len = 1736
pro_no_all = product_data[
    (product_data.a1 == -1) &
    (product_data.a2 == -1) &
    (product_data.a3 == -1)
]                                                      # len = 222
pro_full_info = product_data[
    (product_data.a1 != -1) &
    (product_data.a2 != -1) &
    (product_data.a3 != -1)
]                                                      # len = 17181

######################### 评论数据 #########################

comment_file = 'data/JData_Comment.csv'
comment_data = pd.read_csv(
    comment_file
    # index_col='user_id'
    # encoding='gbk'
)                                  # 558552, 没有缺失值

##### 官方说明 #####
#  dt                   截止到时间         粒度到天
#  sku_id               商品编号           脱敏
#  comment_num          累计评论数分段     0表示无评论
#                                         1表示有1条评论，
#                                         2表示有2-10条评论，
#                                         3表示有11-50条评论，
#                                         4表示大于50条评论
#  has_bad_comment      是否有差评         0表示无，1表示有
#  bad_comment_rate     差评率            差评数占总评论数的比重

##### 属性取值及注释 #####
# dt        从2016-02-01到2016-04-15, 每隔7天统计一次评价, 共12次
# sku_id    共46546种商品, 每种恰好有12条评论
# comment_num       = [0, 1, 2, 3, 4]       数据置信度
# has_bad_comment   = [0, 1]
# bad_comment_rate  = [...]

######################### 用户行为 #########################

action_file_1 = 'data/JData_Action_201602.csv'
action_data_1 = pd.read_csv(
    action_file_1
)                                  # 11485424, 除了model_id, 不存在缺失值
# len(action_data) == len(action_data.drop('model_id', 1).dropna())

##### 官方说明 #####
#  user_id          用户编号
#  sku_id           商品编号
#  time             行为时间, 精确到秒
#  model_id         点击模块编号，如果是点击
#  type	            1.浏览（指浏览商品详情页）；
#                   2.加入购物车
#                   3.购物车删除
#                   4.下单
#                   5.关注
#                   6.点击
#  cate	            品类ID
#  brand	        品牌ID

##### 属性取值及注释 #####
# time                                  可以离散化为一些时间段, 不同时间访问的权重不同
#                                       可能还应该挑出某些节假日
# model_id                              不知道什么意义, 充满了缺失值, 6525807/11485424非空
# type = [1, 2, 3, 4, 5, 6]             说明如上
# cate = [4, 5, 6, 7, 8, 9, 10, 11]     商品明明只有一个cate啊???
# brand                                 可能意义不大

action_file_2 = 'data/JData_Action_201603.csv'
action_data_2 = pd.read_csv(
    action_file_2
)                               # 25916378, 其余同上

action_file_3 = 'data/JData_Action_201604.csv'
action_data_3 = pd.read_csv(
    action_file_3
)                               # 13199934, 其余同上
