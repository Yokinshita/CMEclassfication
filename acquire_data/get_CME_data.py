import requests
from lxml import etree
from datetime import datetime
import re
from tqdm import tqdm
import json
import os


def get_CME_month_datetime_list(year, month):
    # 给定年月，返回该月包含所有CME事件的起止时间和标记的列表
    # 记录每月CME的链接
    CME_month_list_url = 'https://cdaw.gsfc.nasa.gov/CME_list/UNIVERSAL/{0}_{1:0>2d}/univ{0}_{1:0>2d}.html'.format(
        year, month)
    # 记录每日CME的图像的链接
    try:
        month_page = requests.get(CME_month_list_url)
    except Exception:
        print('{} requests fail'.format(CME_month_list_url))
    html = etree.HTML(month_page.text)
    # 查找记录每一次CME的行的tr节点
    month_tr_node_list = html.xpath('//tr//td[@headers="hd_date_time"]/..')
    # 该列表包含了每一个月的CME发生日期、时间以及标注
    print('Processing {0:}/{1:0>2d} data list\n'.format(year, month))
    pbar = tqdm(total=len(month_tr_node_list))
    CME_month_appear_datetime_list = []
    for node in month_tr_node_list:
        # 在每一次事件的Central PA栏中找到标记为Halo的事件，并在本次事件的remark中添加halo
        # 获得每一次CME事件的remarks
        # 若此次事件的Central PA栏中标记为Halo，则在本次事件的remark中添加halo
        is_halo = node.xpath(
            './td[@headers="hd_cpa"]/text()')[0].strip() == 'Halo'
        # *2023年2月3日发现在网页的remarks字段中添加了'<! Remark>\n'，需要去除
        if is_halo:
            remarks = node.xpath('./td[@headers="hd_remark"]/text()')[0].strip(
            ).strip('<! Remark>\n') + 'Halo'
        else:
            remarks = node.xpath('./td[@headers="hd_remark"]/text()')[0].strip(
            ).strip('<! Remark>\n')
        start_end_time_partial_url = node.xpath(
            './td[@headers="hd_date_time"][1]/a/@href')[0]  # 路径前若不加点，则代表向该节点的绝对路径，会出错
        start_end_time_url = '/'.join(CME_month_list_url.split('/')
                                      [0:-1])+'/'+start_end_time_partial_url   # start_end_time_url储存了指向每一次CME事件java movie的链接，希望从中获取起止时刻
        try:
            CME_java_movie_page = requests.get(start_end_time_url)
        except Exception:
            print('\n{} request fail, jump to next CME incident'.format(
                start_end_time_url))
            # 如果请求每次事件网页出错，那么直接跳至下一张图片的tr节点
            continue
        time_regex = re.compile(
            r'.*stime=(\d{8})_(\d{4})&etime=(\d{8})_(\d{4})')  # 匹配起止时间
        result = time_regex.findall(CME_java_movie_page.text)[0]
        # 每次CME事件的开始和结束时间，该时间来源于每次CME的java movie
        CME_start_time = datetime.strptime(result[0]+result[1], r'%Y%m%d%H%M')
        CME_end_time = datetime.strptime(result[2]+result[3], r'%Y%m%d%H%M')

        # 每次CME的发生时间和第一次出现的时刻
        datetime_list = node.xpath('./td[@headers="hd_date_time"]/a/text()')
        CME_month_appear_datetime = datetime.strptime(datetime_list[0].strip()+' ' +
                                                      datetime_list[1].strip(), r'%Y/%m/%d %H:%M:%S')

        CME_datetime_dict = {'appear': CME_month_appear_datetime, 'start': CME_start_time,
                             'end': CME_end_time, 'remark': remarks}  # 记录每次CME事件日期、起始、结束、标记的字典
        CME_month_appear_datetime_list.append(CME_datetime_dict)
        pbar.update(1)
    pbar.close()
    return CME_month_appear_datetime_list


def create_file(path):
    # 判断是否存在path文件夹，若无则创建
    if not os.path.exists(path):
        os.makedirs(path)


def save_json(CME_month_appear_datetime_list, json_filename, encoder):
    # 将包含有每月CME事件的列表储存在json文件
    with open(json_filename, 'w') as f:
        json.dump(CME_month_appear_datetime_list, f, cls=encoder)


class DataEncoder(json.JSONEncoder):
    # 由于json无法序列化datetime类，所以需要添加此类
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.strftime('%Y%m%d %H%M%S')
        else:
            return json.JSONEncoder.default(self, obj)


def save_CME_month_list(CME_month_appear_datetime_list, year, month, save_location):
    create_file(os.path.join(save_location, 'CMElist'))
    json_filename = os.path.join(
        save_location, 'CMElist', '{}_{}_CMEList.json'.format(year, month))
    print('{}/{} data save to {}'.format(year, month, json_filename))
    save_json(CME_month_appear_datetime_list,
              json_filename, DataEncoder)


# if __name__ == '__main__':
#     save_location = r'D:\Programming\CME_data'
#     year, month = 2013, 8
#     CME_month_appear_datetime_list = get_CME_month_datetime_list(year, month)
#     save_CME_month_list(CME_month_appear_datetime_list, year, month,save_location)
