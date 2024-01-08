import  pandas as pd

#与output1合并DataFrame:整合输出
def mergeDf(outputDf1,df2):
    df2['source'] ='eee'
    df2 = df2[['plate','Well','Read 1:600','Read 2:460,510','sample','source']]
    df=outputDf1.append(df2)
    return df
#分组重排数据
def group_rearrangement_df(df,groupColumns=['plate','sample']):
    df1 = pd.DataFrame(columns=df.columns)
    res = df.groupby(groupColumns)
    for index,item in res:
        df1 = df1.append(item)
    df1 = df1.reset_index()
    return df1

# 重新排列df
def rearrange_df(df):
    # 重置索引
    df.index = range(len(df))

    # 重新排列
    li = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    j = 1
    i = 0
    for k in range(len(df)):
        if i > 7:
            j = j + 1
            i = 0
        df.loc[k]['des_well'] = li[i] + f'{j}'
        i = i + 1
    return df

#模糊关键字裁剪无用的列:删除带有read1，read2，well的列
def delNocolumns(df,keyword=['Read','Well','Flu/OD(normal)']):
    columnNames = list(df.columns.values)
    delColumns = []
    for i in range(len(columnNames)):
        for j in range(len(keyword)):
            if keyword[j] in columnNames[i]:
                delColumns.append(columnNames[i])
    df = df.drop(columns=delColumns)
    return df


# 及计算平均值并生成一列
def afterGroupBymean(valiDdata, mediumRead1Means, columnName, plate='plate', read1='Read 1:600'):

    mediumRead1MeansIndex = mediumRead1Means.index.values
    for u in range(len(mediumRead1MeansIndex)):
        tt = list((valiDdata[valiDdata[plate] == mediumRead1MeansIndex[u]][read1]).index)
        for k in range(len(tt)):
            valiDdata.loc[tt[k], columnName] = valiDdata.loc[tt[k], read1] - mediumRead1Means.loc[
                mediumRead1MeansIndex[u]]
    return valiDdata


# 分组求Fluorescence/OD(fin)的平均值、标准差，标准差/平均值，并转换数据排列方式
def changeDataform(valiDdata, columnName='Fluorescence/OD(fin)'):
    #分组重排数据
    valiDdata=group_rearrangement_df(valiDdata)
    # 分组求Fluorescence/OD(fin)的平均值、标准差，标准差/平均值
    print(type(valiDdata))

    pd.to_numeric(valiDdata[columnName], errors ='ignore')
    print(type(valiDdata[columnName][0]))
    feodMean = valiDdata.groupby(['plate', 'sample'])[columnName].mean()
    feodStd = valiDdata.groupby(['plate', 'sample'])[columnName].std()
    feodCv = feodStd / feodMean
    # 统计每一组中'Fluorescence/OD(fin)'的最多个数
    max = (valiDdata.groupby(['plate', 'sample'])[columnName].count()).max()

    df = pd.DataFrame(columns=['plate', 'sample', 'average', 'sd', 'sd/average'], dtype=float)
    # 赋值
    for i in range(max):
        df[f'Fluorescence/OD{i + 1}'] = 0
        df[f'Read 1:600:{i + 1}'] = 0
        df[f'Read 2:460,510:{i + 1}'] = 0
        df[f'Well{i + 1}'] = 0
        df[f'Flu/OD(normal){i + 1}'] = 0

    # 删除无用列
    organizeValiDdata = pd.DataFrame()
    # 'Flu/OD(normal)'
    if ['Read 1:600(normal)', 'Read 2:460,510(normal)'] in list(valiDdata.columns.values):
        organizeValiDdata = valiDdata.drop(columns=
                                           ['Read 1:600(normal)', 'Read 2:460,510(normal)']
                                           )
    else:
        organizeValiDdata = valiDdata
    # 分组
    counts = organizeValiDdata.groupby(['plate', 'sample']).count()
    ps = list(counts.index.values)
    for i in range(len(ps)):
        df.loc[i, 'plate'] = ps[i][0]
        df.loc[i, 'sample'] = ps[i][1]
    # 重置索引
    organizeValiDdata.index = range(len(organizeValiDdata))
    # 设置元组索引
    df.index = [x for x in zip(df['plate'], df['sample'])]
    # 转换类型
    bcd = pd.Series(counts['Fluorescence/OD(fin)'].values, index=list(counts.index.values))
    k = 1
    for i in range(len(organizeValiDdata)):
        aa = tuple([organizeValiDdata.loc[i, 'plate'], organizeValiDdata.loc[i, 'sample']])
        df[f'Fluorescence/OD{k}'].loc[[aa]] = organizeValiDdata.loc[i, 'Fluorescence/OD(fin)']
        df[f'Flu/OD(normal){k}'].loc[[aa]] = organizeValiDdata.loc[i, 'Flu/OD(normal)']
        df[f'Read 1:600:{k}'].loc[[aa]] = organizeValiDdata.loc[i, 'Read 1:600']
        df[f'Read 2:460,510:{k}'].loc[[aa]] = organizeValiDdata.loc[i, 'Read 2:460,510']
        df[f'Well{k}'].loc[[aa]] = organizeValiDdata.loc[i, 'Well']

        if k == bcd[aa]:
            k = 1
        else:
            k = k + 1
    for item in list(df.index.values):
        df['average'].loc[[item]] = feodMean.loc[item]
        df['sd'].loc[[item]] = feodStd.loc[item]
        df['sd/average'].loc[[item]] = feodCv.loc[item]
    return df


def reverseChangeDataform(dataDf):
    df = pd.DataFrame(
        columns=['plate', 'sample', 'Well', 'Read 1:600', 'Read 2:460,510', 'Fluorescence/OD(fin)', 'Flu/OD(normal)'])
    columnNames = list(dataDf.columns.values)
    dataDf.index = range(len(dataDf))
    feodCount = (len(columnNames) - 5) / 5

    j = 0
    k = 1
    for i in range(len(dataDf)):

        if k > feodCount:
            k = 1
        while k <= feodCount:
            df.loc[j, 'plate'] = dataDf.loc[i, 'plate']
            df.loc[j, 'sample'] = dataDf.loc[i, 'sample']
            df.loc[j, 'Fluorescence/OD(fin)'] = dataDf.loc[i, f'Fluorescence/OD{k}']
            df.loc[j, 'Flu/OD(normal)'] = dataDf.loc[i, f'Flu/OD(normal){k}']
            df.loc[j, 'Well'] = dataDf.loc[i, f'Well{k}']
            df.loc[j, 'Read 1:600'] = dataDf.loc[i, f'Read 1:600:{k}']
            df.loc[j, 'Read 2:460,510'] = dataDf.loc[i, f'Read 2:460,510:{k}']
            j = j + 1
            k = k + 1

    return df




