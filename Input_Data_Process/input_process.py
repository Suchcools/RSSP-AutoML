import pandas as pd
def read_excel(inputSampleDataFilePath, inputSampleNameFilePath):
    df_data_dic = {}
    df_dscription_dic = {}
    df_list = [df_data_dic, df_dscription_dic]

    inputSampleDataFile = pd.ExcelFile(inputSampleDataFilePath)
    inputSampleNameFile = pd.ExcelFile(inputSampleNameFilePath)
    print(inputSampleDataFile.sheet_names, inputSampleNameFile.sheet_names)

    for s_name in inputSampleDataFile.sheet_names:
        inputSampleDataDf = pd.read_excel(inputSampleDataFilePath, sheet_name=s_name)
        inputSampleNameDf = pd.read_excel(inputSampleNameFilePath, sheet_name=s_name)

        # 取索引
        index = inputSampleDataDf.index[inputSampleDataDf[inputSampleDataDf.columns.values[1]] == 'Well'].tolist()
        # 取所在的行
        row = inputSampleDataDf[inputSampleDataDf[inputSampleDataDf.columns.values[1]].isin(['Well'])]
        inputSampleDataDf.loc[index]
        index = int(index[0])
        print(index)
        # 软件运行说明
        softwareDescription = inputSampleDataDf.loc[:index - 1]
        # 去掉列NAN
        new_inputSampleDataDf = inputSampleDataDf.loc[index + 1:].dropna(axis=1, how='any', inplace=False)

        # 重置索引
        new_inputSampleDataDf = new_inputSampleDataDf.reset_index(drop=True)
        new_inputSampleDataDf
        # 获取列的名字
        new_inputSampleDataDf_columnsNames = new_inputSampleDataDf.columns.values
        inputSampleNameDf_columnsNames = inputSampleDataDf.columns.values

        # 重命名 new_inputSampleDataDf
        new_inputSampleDataDf.rename({new_inputSampleDataDf_columnsNames[0]: row.loc[index][row.columns.values[1]],
                                      new_inputSampleDataDf_columnsNames[1]: row.loc[index][row.columns.values[2]],
                                      new_inputSampleDataDf_columnsNames[2]: row.loc[index][row.columns.values[3]]},
                                     axis='columns', inplace=True)

        # 重命名 inputSampleNameDf
        inputSampleNameDf.columns = ['Well', 'sample']
        inputSampleNameDf.rename({inputSampleNameDf_columnsNames[0]: 'Well',
                                  inputSampleNameDf_columnsNames[1]: 'sample'}, axis='columns', inplace=True)

        # 合并
        beginDf = pd.merge(new_inputSampleDataDf, inputSampleNameDf)
        # 加一列
        beginDf['plate'] = s_name
        # 列排序
        beginDf = beginDf[['plate', 'Well', 'Read 1:600', 'Read 2:460,510', 'sample']]

        beginDf.dropna(axis=0, how='any', inplace=True)

        # 加入df
        dic_data_item = {s_name: beginDf}
        df_data_dic.update(dic_data_item)
        dic_description_item = {s_name: softwareDescription}
        df_dscription_dic.update(dic_description_item)
    return df_list


def write_to_BeginExcel(df_list,inputSampleDataFilePath):
    inputSampleDataFile = pd.ExcelFile(inputSampleDataFilePath)
    # 输出excel
    for i in range(len(df_list)):
        w = pd.ExcelWriter(f'beginExcel{i}.xlsx')
        for s_name in inputSampleDataFile.sheet_names:
            if i == 1:
                df_list[i].get(s_name).to_excel(w, sheet_name=s_name, index=False, header=False)
            else:
                df_list[i].get(s_name).to_excel(w, sheet_name=s_name, index=False, header=True)
        w.save()


if __name__ == '__main__':
    inputSampleDataFilePath = 'E:/code/input_excelFile/input-SampleDataFile.xlsx'
    inputSampleNameFilePath = 'E:/code/input_excelFile/input-SampleNameFile.xlsx'
    # 读取文件
    df_list = read_excel(inputSampleDataFilePath, inputSampleNameFilePath)
    # 写出文件
    write_to_BeginExcel = write_to_BeginExcel(df_list, inputSampleDataFilePath)

