#!/usr/bin/env python
# coding: utf-8

# In[1]:


from iFinDPy import *
import tushare as ts
import pandas as pd
import numpy as np
from tqdm import tqdm
import pytz
from datetime import datetime
from datetime import timedelta
import time
from itertools import groupby
from scipy.stats import norm


# In[9]:


"""
中国上市的所有期权最近3期的实时数据
包括期货期权和股指期权
"""

def cal_prob(price_array, window, threshold):
    """
    此函数计算在一定窗口期内，股价变动大于或小于一定水平线的概率
    若thershold<0, 获得股价在一段时间后小于一定水平线的概率
    若thershold>0, 获得股价在一段时间后大于一定水平线的概率
    要先删除其中不正常的0值点
    """
        
    change = []
    while float(0) in price_array:
        price_array.remove(float(0))
    
    for i in range(len(price_array)-window):
        subarray = price_array[i:i+window+1]
        change.append(subarray[-1]/subarray[0]-1)
    if threshold>0:
        prob = np.sum(np.array(change)>=threshold)/len(change)
    else:
        prob = np.sum(np.array(change)<=threshold)/len(change)
    return prob






class Ifind_option():
    
    def __init__(self, exchanges = ['DCE','CZCE','SHFE','INE','SSE']):
        """
        输入交易所列表
        默认交易所为中国所有具有期权交易的交易所
        """

        self.ex_l = exchanges
    
    def gettime(self):
        """
        获取当前时刻的北京时间
        保存为日期(年月日)和时点(年月日分钟秒)
        不同的api需要不用的日期格式
        """

        tz = pytz.timezone('Asia/Shanghai') #东八区
        ifdate = datetime.fromtimestamp(int(time.time()),
                                        pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d')
        tsdate= datetime.fromtimestamp(int(time.time()),
                                       pytz.timezone('Asia/Shanghai')).strftime('%Y%m%d')
        yesterday = datetime.fromtimestamp(int(time.time()),
                                       pytz.timezone('Asia/Shanghai'))- timedelta(days=1)
        t = datetime.fromtimestamp(int(time.time()),
                                   pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')
        
        self.tsdate = tsdate
        self.ifdate = ifdate
        self.now = t
        self.yes = yesterday.strftime('%Y-%m-%d')
        print("今天日期：", self.ifdate)
        print("现在时点：", self.now)
        
    def pool(self):
        """
        获取在目标多个交易所内的所有期权代码
        """
        
        print("正在获取期权池:")
        all_options = pd.DataFrame()
        for exchange in self.ex_l:
            all_options = all_options.append(pro.opt_basic(exchange=exchange, 
                                                       fields='ts_code,name,maturity_date,exercise_price,exercise_type,call_put,exchange'))
        ## 实时选取还未到期的期权并加入标的物类型
        valid_options = all_options[all_options['maturity_date']>self.tsdate].copy()
        valid_options['underlying'] = [valid_options.iloc[i]['name'].split('期权')[0] for i in range(len(valid_options))]
      
        ## 得到所有标的物类型list
        underlying_list = list(set(valid_options['underlying']))

        ## 对所有标的物类型，获取最近快到期三期的期权数据, 只需记录ts_code,strike和类型(put or call)。
        tscode_list = []
        strike_list = []
        type_list = []
        exchange_list = []
        maturity_list = []
        underlying_list2 = []
        for underlying in tqdm(underlying_list):
            subdata = valid_options[valid_options['underlying'] == underlying]
            ## 获取最近三期, 可能最终只有1,2个有效日期，3个以上选取3个
            date_list = list(set(subdata['maturity_date']))
            date_list.sort()
            subdata2 = pd.DataFrame()
            for date in date_list[:3]:
                subdata2 = subdata2.append(subdata[subdata['maturity_date'] == date])
        
            tscode_list += list(subdata2['ts_code'])
            strike_list += list(subdata2['exercise_price'])
            type_list += list(subdata2['call_put'])
            exchange_list += list(subdata2['exchange'])
            maturity_list += list(subdata2['maturity_date'])
            underlying_list2 += list(subdata2['underlying'])
            
        ## 将tscode转换为ifcode
        ifcode_list = []
        for tscode in tscode_list:
            split = tscode.split(".")
            if split[1] == 'ZCE':
                ifcode_list.append(split[0]+'.'+'CZC')
            elif split[1] == 'CFX':
                ifcode_list.append(split[0]+'.'+'CFE')
            else:
                ifcode_list.append(tscode)
    
        ## 赋值给期权池特征
        self.pool = pd.DataFrame()
        self.pool['ts_code'] = tscode_list
        self.pool['if_code'] = ifcode_list
        self.pool['行权价'] = strike_list
        self.pool['call_put'] = type_list
        self.pool['exchange'] = exchange_list 
        self.pool['到期日'] = maturity_list
        self.pool['underlying'] = underlying_list2
        self.underlying = underlying_list 
        print("商品期权种类共计{0}种，期权池共计{1}个。\n".format(len(underlying_list),len(tscode_list)))
        print("标的物有：", underlying_list)
        return self.pool
    
    
    def future_info(self):
        """
        获取期权池中各个期权对应的标的物信息
        期货期权的标的物为期货
        股指期权的标的物为各种ETF
        """
        
        print("正在获取各期权标的物的代码：")
        underlying_code_list = []
        ## 各个交易所的期权代码格式不同，分开处理
        for i in range(len(self.pool)):
            option = self.pool.iloc[i]
            
            if option['exchange'] == 'CFFEX':
                ## 中金所只交易沪深300的期权
                underlying_code_list.append('000300.SH')
                
            if option['exchange'] == 'SSE':
                ## SSE只交易华夏50ETF和华夏300ETF
                if option['underlying'] == '华夏上证50ETF':
                    underlying_code_list.append('510050.OF')
                if option['underlying'] == '华泰柏瑞沪深300ETF':
                    underlying_code_list.append('510300.OF')
            
            if option['exchange'] == 'DCE':
                underlying_code_list.append(option['ts_code'].split('-')[0] + '.' + option['ts_code'].split('.')[-1])
            
            if option['exchange'] == 'SHFE':
                split = [''.join(g) for _, g in groupby(option['ts_code'],
                                                        key=lambda x: x.isdigit()*'d' or x.isalpha()*'a' )]
                underlying_code_list.append(split[0]+split[1]+split[-2]+split[-1])
        
            if option['exchange'] == 'CZCE':
                split = [''.join(g) for _, g in groupby(option['ts_code'],
                                                        key=lambda x: x.isdigit()*'d' or x.isalpha()*'a' )]
                underlying_code_list.append(split[0] + split[1] + split[-2] + 'CZC')
        
        self.pool['标的物代码'] = underlying_code_list
        ## 为了Ifind的顺利爬取，将标的物的code进行字符的合并，以','分隔不同的标的物code
        underlying_code_str = ''
        for underlying_code in list(set(underlying_code_list)): 
            underlying_code_str = underlying_code_str + underlying_code + ','
        underlying_code_str = underlying_code_str[:-1]
        
        ## 利用拼接的字符串爬取标的物的现价
        print("正在获取各期权标的物的实时价格表：")
        underlying_price_table = pd.DataFrame()
        result = THS_RealtimeQuotes(underlying_code_str, 'preClose')['tables']
        underlying_price_table['标的物'] =  [d['thscode'] for d in result]
        underlying_price_table['实时价格']= [d['table']['preClose'][0] for d in result]
        underlying_price_dict = dict(zip(underlying_price_table['标的物'], underlying_price_table['实时价格']))
        
        ## 通过对表的形式将给予pool表新列“标的物价格”
        underlying_price_list = []
        invalid_underlying_list  = []
        for underlying_code in underlying_code_list:
            try:
                underlying_price_list.append(underlying_price_dict[underlying_code])
            except:
                underlying_price_list.append(np.nan)
                invalid_underlying_list.append(underlying_code)
                continue
                
        print("以下标的物的数据缺失：", list(set(invalid_underlying_list)))
        self.pool['标的物价格'] = underlying_price_list 
            
        
    def option_info(self):
        """
        通过Ifind实时接口爬取期权的期权价, 历史波动率
        进行深度价外的期权的选择
        """
        
        print("正在爬取期权的期权价和历史波动率:")
        ## 为了Ifind的顺利爬取，将标的物的code进行字符的合并，以','分隔不同的期权code
        option_code_str = ''
        for option_code in self.pool['if_code']: 
            option_code_str = option_code_str + option_code + ','
        option_code_str = option_code_str[:-1]
        
        result = THS_RealtimeQuotes(option_code_str, 'preClose;historyVolatility')['tables']
        self.pool['期权价'] =  [d['table']['preClose'][0] for d in result]
        self.pool['历史波动率']= [d['table']['historyVolatility'][0] for d in result]
        
        ## --------------------------- 计算离到期日的时间(单位：年) -------------------------- ##
        T_list = []
        for i in range(len(self.pool)):
            option = self.pool.iloc[i]
            maturity = datetime.strptime(str(option['到期日']),"%Y%m%d")
            today = datetime.strptime(str(self.tsdate),"%Y%m%d")
            T_list.append((maturity-today).days/365)
        self.pool['T'] = T_list
        
        ## ------------------------------------- 获取短期国债利率 --------------------------- ##
        self.r = pro.us_tltr(start_date='20200101', end_date=self.tsdate, fields='ltc').iloc[0][0]/100
        
        
        
        ## ----------------------------- 获取深度价外期权（虚值）列表 ----------------------- ##
        print("正在选取深度价外期权：")
        ## 利用(ln(ST)-ln(S0)-(r+0.5*sigma^2)T/sigma*sqrt(T)  ~ N(0,1)
        ## 先大致用BS法选取深度价外期权, 范围可以大一些，后边再用overlapping time window来过滤
        ## 并且用BS法计算行权概率
        
        self.valid_pool = pd.DataFrame()
        self.pool.dropna(inplace =True)
        for i in range(len(self.pool)):
            try:
                option = self.pool.iloc[i]
            
                ## 对看涨看跌期权分情况考虑
                sigma = option['历史波动率']
                T = float(option['T'])
                r = self.r
                d1 = (np.log(option['标的物价格']/option['行权价'])+(r + sigma**2/2)*T)/(sigma*np.sqrt(T))
                d2 = d1 - sigma*np.sqrt(T)
                N = norm.cdf
            
                if option['call_put'] == 'C':
                    if N(d2) < 0.08:
                        self.valid_pool = self.valid_pool.append(option)
                if option['call_put'] == 'P':
                    if N(-d2) < 0.08:
                        self.valid_pool = self.valid_pool.append(option)
            
            except:
                continue
        self.valid_pool.index = range(len(self.valid_pool))
        
        ## -------------------------- 保证金爬取 ---------------------------------------------- ##
        print("正在获得期权保证金和合约乘子：")
        ## 判断若不在交易时间，则使用昨天的保证金，不然是na值
        ## 拿一个期权测试一下
        test_result = THS_BasicData(self.valid_pool.iloc[0]['if_code'],
                                    'ths_td_unit_unit_option;ths_trade_deposit_option',
                                     self.ifdate + ';' + self.ifdate)['tables'][0]['table']['ths_trade_deposit_option'][0]
        if test_result == None:
            choosed_date = self.yes
        else:
            choosed_date = self.ifdate
        valid_option_code_str = ''
        for valid_option_code in self.valid_pool['if_code']: 
            valid_option_code_str = valid_option_code_str + valid_option_code + ','
        valid_option_code_str = valid_option_code_str[:-1]
        result = THS_BasicData(valid_option_code_str,
                               'ths_td_unit_unit_option;ths_trade_deposit_option',
                               choosed_date + ';' + choosed_date)['tables']
        self.valid_pool['合约乘子'] =  [d['table']['ths_td_unit_unit_option'][0] for d in result]
        self.valid_pool['期权保证金']= [d['table']['ths_trade_deposit_option'][0] for d in result]
        
        ## ------------------------------ 全损点计算 ------------------------------------------ ##
        ## 全损点即为初始保证金和期权费全部损失的价格点
        game_over_point_list = []
        for i in range(len(self.valid_pool)):
            option = self.valid_pool.iloc[i]
            multiplier = option['合约乘子']
            K = option['行权价']
            S0 = option['标的物价格']
            p = option['期权价']
            margin = option['期权保证金']
            if option['call_put'] == 'C':
                game_over_point = (margin+multiplier*p)/multiplier + K
            else:
                game_over_point = K - (margin+multiplier*p)/multiplier
            game_over_point_list.append(game_over_point)
        
        self.valid_pool['全损点'] = game_over_point_list
        
        
        ## ---------------------------- 被行权概率计算（BS法） ------------------------------------------  ##
        exe_prob_list = []
        for i in range(len(self.valid_pool)):
            try:
                option = self.valid_pool.iloc[i]
            
                ## 对看涨看跌期权分情况考虑
                sigma = option['历史波动率']
                T = float(option['T'])
                r = self.r
                d1 = (np.log(option['标的物价格']/option['行权价'])+(r + sigma**2/2)*T)/(sigma*np.sqrt(T))
                d2 = d1 - sigma*np.sqrt(T)
                N = norm.cdf
            
                if option['call_put'] == 'C':
                    exe_prob_list.append(N(d2))
                if option['call_put'] == 'P':
                    exe_prob_list.append(N(-d2))
            
            except:
                continue
        
        
        self.valid_pool['被行权概率(BS法)'] = exe_prob_list
        
        ## ----------------------------- 全损概率计算（BS法） --------------------------------------------- ##
        
        lose_prob_list = []
        for i in range(len(self.valid_pool)):
            try:
                option = self.valid_pool.iloc[i]
            
                ## 对看涨看跌期权分情况考虑
                sigma = option['历史波动率']
                T = float(option['T'])
                r = self.r
                d1 = (np.log(option['标的物价格']/option['全损点'])+(r + sigma**2/2)*T)/(sigma*np.sqrt(T))
                d2 = d1 - sigma*np.sqrt(T)
                N = norm.cdf
            
                if option['call_put'] == 'C':
                    lose_prob_list.append(N(d2))
                if option['call_put'] == 'P':
                    lose_prob_list.append(N(-d2))
            
            except:
                continue
        self.valid_pool['全损概率(BS法)'] = lose_prob_list
        
        
    
    def history_prob(self):
        """
        获取各个标的物的历史价格，为了概率的计算
        """
        
        print("正在获取标的物历史信息：")
        ## 用字典记录
        underlying_history_diction = dict()
        for underlying_code in list(set(self.valid_pool['标的物代码'])):
            underlying_history_diction[underlying_code] = THS_HistoryQuotes(underlying_code,
                                                                            'close',
                                                                            'Interval:D,CPS:1,baseDate:1900-01-01,Currency:YSHB,fill:Previous',
                                                                            '2015-01-01',
                                                                            self.ifdate)['tables'][0]['table']['close']
        
        exe_prob_list = []
        ## 为所有深度价外期权计算被行权概率（历史法）
        for i in range(len(self.valid_pool)):
            option = self.valid_pool.iloc[i]
            threshold = option['行权价']/option['标的物价格'] - 1
            underlying_history = underlying_history_diction[option['标的物代码']]
            window = int(option['T']*252)
            exe_prob_list.append(cal_prob(underlying_history, window, threshold))
        
        self.valid_pool['被行权概率(历史法)'] = exe_prob_list
            
        
        lose_prob_list = []
        ## 计算期权保证金全损失概率（历史法）
        for i in range(len(self.valid_pool)):
            option = self.valid_pool.iloc[i]
            threshold = option['全损点']/option['标的物价格'] - 1
            underlying_history = underlying_history_diction[option['标的物代码']]
            window = int(option['T']*252)
            lose_prob_list.append(cal_prob(underlying_history, window, threshold))
        
        self.valid_pool['全损概率(历史法)'] = lose_prob_list        
        
    def cal_return(self):
        """
        计算各个期权的收益率，包括BS方法计算的收益率和历史法计算的收益率
        分为年化和不年化的
        收益率 = (期望获利) /保证金
        期望获利 = (1-被行权概率)*乘子*期权费  - 全损概率*保证金
        """
        
        BS_return_list = []
        his_return_list = []
        BS_annual_return_list = []
        his_annual_return_list = []
        for i in range(len(self.valid_pool)):
            option = self.valid_pool.iloc[i]
            T = option['T']
            margin = option['期权保证金']
            multiplier = option['合约乘子']
            p = option['期权价']
            BS_return = ((1-option['被行权概率(BS法)'])*multiplier*p - option['全损概率(BS法)']*margin)/margin
            his_return = ((1-option['被行权概率(历史法)'])*multiplier*p - option['全损概率(历史法)']*margin)/margin
            BS_annual_return = (1+BS_return)**(1/T) - 1
            his_annual_return = (1+his_return)**(1/T) - 1
            BS_return_list.append(BS_return)
            his_return_list.append(his_return)
            BS_annual_return_list.append(BS_annual_return)
            his_annual_return_list.append(his_annual_return)
        
        self.valid_pool['BS法收益率'] = BS_return_list
        self.valid_pool['历史法收益率'] = his_return_list
        self.valid_pool['BS年化收益率'] = BS_annual_return_list
        self.valid_pool['历史法年化收益率'] = his_annual_return_list
    
    def cal_greek(self):
        """
        Greek使用BS法计算
        """
        
        delta_list = []
        gamma_list = []
        vega_list = []
        theta_list = []
        rho_list = []
        
        print("正在计算各个期权的Greek：")
        for i in range(len(self.valid_pool)):
            option = self.valid_pool.iloc[i]
            r = self.r
            sigma = option['历史波动率']
            K = option['行权价']
            T = option['T']
            S0 = option['标的物价格']
            
            N = norm.cdf
            M = norm.pdf
            d1 = (np.log(S0/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option['call_put'] == 'C':
                delta_list.append(round(N(d1),4))
                gamma_list.append(round(M(d1)/(sigma*S0*np.sqrt(T)),4))
                vega_list.append(round(S0*M(d1)*np.sqrt(T)*0.01,4))
                theta_list.append(round((- S0*M(d1)*sigma/(2*np.sqrt(T)) - r*K*np.exp(-r*T)*N(d2))/365,4))
                rho_list.append(round(K*T*np.exp(-r*T)*N(d2)/100,4))
        
            else:
                delta_list.append(round(N(d1) - 1, 4))
                gamma_list.append(round(M(d1)/(sigma*S0*np.sqrt(T)),4))
                vega_list.append(round(S0 * M(d1) * np.sqrt(T) * 0.01,4))
                theta_list.append(round((- S0*M(d1)*sigma/(2*np.sqrt(T)) + r * K * np.exp(-r*T)*N(-d2))/365,4))
                rho_list.append(round(-K*T*np.exp(-r*T)*N(-d2)/100,4))
        
        self.valid_pool['Delta'] = delta_list
        self.valid_pool['Gamma'] = gamma_list
        self.valid_pool['Vega'] = vega_list
        self.valid_pool['Theta'] = theta_list
        self.valid_pool['rho'] = rho_list
        
        
    
    def save_to_excel(self):
        """
        用excel记录结果
        不同sheet记录不同类型标的物
        只取有用的列
        """
        print("正在保存至excel.")
        writer = pd.ExcelWriter('深度虚值期权表.xlsx')
        underlying_list = list(set(self.valid_pool['underlying']))
        for underlying in underlying_list:
            subdata = self.valid_pool[self.valid_pool['underlying'] == underlying][['if_code','underlying','标的物代码','call_put',
                                                                                    '到期日','行权价','期权价','被行权概率(BS法)','被行权概率(历史法)',
                                                                                    '全损概率(BS法)', '全损概率(历史法)',
                                                                                    'BS法收益率','历史法收益率',
                                                                                    'BS年化收益率','历史法年化收益率',
                                                                                    'Delta','Gamma','Vega','Theta','rho']]
            result = subdata.sort_values(by = ['到期日','call_put','行权价'], ascending=[True, True, True])
            result.index = range(len(result))
            result.to_excel(writer, underlying, encoding = 'utf_8_sig')
            
        writer.save()
        
        
            
if __name__ == '__main__':

    ## 个人信息设置
    THS_iFinDLogin('xfrs068','883359')
    wfc_ts_token = '72a3fb11aa9d09b4dbe8019bc6925c5319c27fff1c8a3e5715c72006'
    ts.set_token(wfc_ts_token)
    pro = ts.pro_api(wfc_ts_token)

    ## 基本数据爬取
    op = Ifind_option()
    op.gettime()
    op.pool()
    op.future_info()
    op.option_info()
    op.history_prob()
    op.cal_return()
    op.cal_greek()
    op.save_to_excel()


# In[ ]:




