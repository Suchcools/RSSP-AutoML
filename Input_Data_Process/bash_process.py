from pydoc import source_synopsis
import pandas as pd
import utils as util
import dealWithJudges as jds
import warnings
warnings.filterwarnings('ignore')
import time
import os
date_time= time.strftime('%Y_%m_%d', time.localtime())
source_path = f'./code/output_excelFile/{date_time}'

# 第一条件不符合，输出df
def output_df1(beginExcelFilePath, columns):
    beginExcelFile = pd.ExcelFile(beginExcelFilePath)
    df1 = pd.DataFrame(columns=columns)
    df2 = pd.DataFrame(columns=columns)
    for s_name in beginExcelFile.sheet_names:
        # 进入第一个判断
        outputDf = jds.dealWith_judge1(beginExcelFilePath,s_name)
        # outputDf[0]:不符合第一个条件的df
        # outputDf[1]:符合第一条件的df
        df1 = df1.append(outputDf[0])
        df2 = df2.append(outputDf[2])

    df1['source'] = 'slow'
    df2['source'] = 'slow'
    df2['reason'] = 'read1＜0.15'
    df2.to_excel(f'{source_path}/{date_time}_noValidData1.xlsx',index=False)
    return df1



# 第二条件不符合，输出df
def output_df2(beginExcelFilePath, columns):

    beginExcelFile = pd.ExcelFile(beginExcelFilePath)
    df1 = pd.DataFrame(columns=columns)
    df2 = pd.DataFrame()
    for s_name in beginExcelFile.sheet_names:
        # 进入第一个判断
        validData_dfOneTrue = jds.dealWith_judge1(beginExcelFilePath,s_name)[1]
        # 进入第二个判断
        outputDf = jds.dealWith_judge2(validData_dfOneTrue)
        # outputDf[0]:不符合第二个条件的df
        # outputDf[1]:符合第二条件的df
        df1 = df1.append(outputDf[0])
        df2 = df2.append(outputDf[2])

    # 给列重命名并且赋值
    df1.columns = ['Source_plate', 'Source_well']
    df1['Des_plate'] = 'flu-plate2'
    df1['des_well'] = '12'
    df1['Vol'] = '150'

    # 重新排列
    df1 = util.rearrange_df(df1)
    df2['reason']='OVRFLW'
    df2.to_excel(f'{source_path}/{date_time}noValidData2.xlsx',index=False)
    return df1

# 第三条件不符合，输出df
def output_df3(beginExcelFilePath, columns):
    beginExcelFile = pd.ExcelFile(beginExcelFilePath)
    df = pd.DataFrame(columns=columns)
    for s_name in beginExcelFile.sheet_names:
        # 进入第一个判断
        validData_dfOneTrue = jds.dealWith_judge1(beginExcelFilePath,s_name)[1]
        # 进入第二个判断
        validData_dfTwoTrue = jds.dealWith_judge2(validData_dfOneTrue)[1]
        # 进入第三个判断
        outputDf = jds.dealWith_judge3(validData_dfTwoTrue)
        # outputDf[0]:不符合第一个条件的df
        # outputDf[1]:符合第一条件的df
        df = df.append(outputDf[0])

    # 删除第三结果不需要的列
    df = df.drop(['Read 1:600', 'Read 2:460,510', 'sample'], axis=1)
    # 列重名名
    df.columns = ['Source_plate', 'Source_well']  # ,'Des_plate','des_well','Vol'
    df['Des_plate'] = 'flu-plate2'
    df['des_well'] = '12'
    df['Vol'] = '150'

    # 重新排列
    df = util.rearrange_df(df)
    return df


# 生成有效处理文件
def product_valiDdata(beginExcelFilePath, columns):
    beginExcelFile = pd.ExcelFile(beginExcelFilePath)
    df = pd.DataFrame(columns=columns, dtype=float)
    mediumsDf = pd.DataFrame(columns=columns, dtype=float)
    for s_name in beginExcelFile.sheet_names:
        # 进入第一个判断
        validData_dfOneTrue = jds.dealWith_judge1(beginExcelFilePath,s_name)[1]
        # 得到medium数据
        medium_validData_dfOneTrue = jds.dealWith_judge1(beginExcelFilePath,s_name)[3]
        # 进入第二个判断
        validData_dfTwoTrue = jds.dealWith_judge2(validData_dfOneTrue)[1]
        # 进入第三个判断
        outputDf = jds.dealWith_judge3(validData_dfTwoTrue)
        # outputDf[0]:不符合条件的df
        # outputDf[1]:符合条件的df
        df = df.append(outputDf[1])
        # 加入medium
        mediumsDf = mediumsDf.append(medium_validData_dfOneTrue)

    return df, mediumsDf

#输出重新转化的整合
def output_df4(df):
    #执行一些列操作
    valiDdata = jds.dealWith_action(df[0],df[1])

    dataDf = util.changeDataform(valiDdata)
    # print(dataDf[:10])
    #第四个判断:是否小于0.3且大于0
    listFour1 = jds.dealWith_judge4(dataDf) #li[0]:是  li[1]：否
    #是小于0.3，删除无用列
    vdOutput1= util.delNocolumns(listFour1[0],keyword=['Read','Well','Flu/OD(normal)'])
    pd.set_option('display.max_columns',None) #显示完整

     #不小于0.3 ，进入第五个判断和1000比较：是否小于1000
    listFive = jds.dealWith_judge5(util.reverseChangeDataform(listFour1[1])) #li[0]:是小于1000  li[1]:否小于1000

    return (vdOutput1,listFive)



def output_df5(df):

    #变换数据格式  变为一组一行,且计算'Flu/OD(normal)'的平均值...
    df = util.changeDataform(df,columnName='Flu/OD(normal)')
    #第四个判断:是否小于0.3且大于0
    li =jds.dealWith_judge4(df) #li[0]:是  li[1]：否
    #是0<XX<0.3，删除无用列  同时输出有效数据
    vdOutput1= util.delNocolumns(li[0],keyword=['Read','Well','Fluorescence/OD'])
    #不是0<XX<0.3 li[1]
    df1 = util.reverseChangeDataform(li[1])
     #删除表中含有任何NaN的行
    df = df.dropna(axis=0,how='any')
    return vdOutput1,df1


def write_output_excel(dfTwo, dfThree, dfFour, dfFive1, dfFive2):

    dfTwo.to_csv(f'{source_path}/{date_time}_Dilution_determination_output.csv', index=False)
    dfThree.to_csv(f'{source_path}/{date_time}_continue_cultivate_output.csv', index=False)

    dfFour.to_excel(f'{source_path}/{date_time}_reconvert_output.xlsx', index=False)
    # 输出一个Excel多个sheet
    w = pd.ExcelWriter(f'{source_path}/{date_time}_validData_output.xlsx')
    dfFive = [dfFive1, dfFive2]
    for i in range(len(dfFive)):
        dfFive[i].to_excel(w, sheet_name=f'sheet{i}', index=False)  
    w.save()

def del_no_row(dfFive2):
    # 删除无用列
    for index, row in dfFive2.iterrows():
        if row.count() > 7:
            continue
        else:
            dfFive2.drop(index, axis=0, inplace=True)
    return dfFive2

if __name__ == '__main__':

    # 读取文件地址
    beginExcelFilePath = 'beginExcel0.xlsx'
    if not os.path.exists(f'./output_excelFile/{date_time}'):  # os模块判断并创建
        os.mkdir(f'./output_excelFile/{date_time}')
    # 第一个输出结果
    dfOne = output_df1(beginExcelFilePath,
                       columns=['plate', 'Well', 'Read 1:600', 'Read 2:460,510', 'sample', 'source'])


    # 第二个输出结果
    dfTwo = output_df2(beginExcelFilePath, columns=['plate', 'Well'])



    # 第三个输出结果
    dfThree = output_df3(beginExcelFilePath, columns=['plate', 'Well'])
    # 生成有效数据 vd：vd[0]:不含medium，vd[1]:只含有medium
    vd = product_valiDdata(beginExcelFilePath,
                           columns=['plate', 'Well', 'Read 1:600', 'Read 2:460,510', 'sample'])

    # 第四个输出结果:直接输出有效数据tt[0],tt[1]输出和1000比较的结果列表:(li[0]:是小于1000  li[1]:否小于1000)
    tt = output_df4(vd)
    dfFive1 = tt[0]
    # 不小于1000，和之前数据整合输出：
    dfFour = util.mergeDf(dfOne, tt[1][1])
    # 第五个输出结果
    # dfFive[0]需要进一步判断，dfFive[1]直接输出有效数据成为一个sheet
    dfFour = dfOne
    dfFive2 = pd.DataFrame()
    if len(tt[1][0]) != 0:
        cc = output_df5(tt[1][0])
        # 直接输出有效数据
        dfFive2 = cc[0]
        #删除无用列
        dfFive2 = del_no_row(dfFive2)
        dfFour = util.mergeDf(dfOne, cc[1])
    #写出excel
    write_output_excel(dfTwo,dfThree,dfFour,dfFive1,dfFive2)

