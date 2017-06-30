# """
# Created on Tue Nov 10 14:23:21 2015
#
# @author: Administrator
# """
#
# #读取excel使用(支持03)
# import xlrd
# #写入excel使用(支持03)
# #import xlwt3
#
#
# #读取execel使用(支持07)
# from openpyxl import Workbook
# #写入excel使用(支持07)
# from openpyxl import load_workbook
#
# def showexcel(path):
#     workbook=xlrd.open_workbook(path)
#     sheets=workbook.sheet_names();
#     #多个sheet时，采用下面的写法打印
#     for sname in sheets:
#         print(sname)
#     worksheet=workbook.sheet_by_name(sheets[0])
#     #nrows=worksheet.nrows
#     #nclows=worksheet.ncols
#     for i in range(0,worksheet.nrows):
#         row=worksheet.row(i)
#         for j in range(0,worksheet.ncols):
#             print(worksheet.cell_value(i,j),type(worksheet.cell_value(i,j)),"\t",end="")
#
#
#         print(end="\n")
#
# def writeexcel07(path,tableValue):
#
#     wb=Workbook()
#     #sheet=wb.add_sheet("xlwt3数据测试表")
#     sheet=wb.create_sheet("Sheet1",0)
#     rowNum=len(tableValue)
#     columNum=len(tableValue[0])
#     print("rowNum",rowNum,"--rowNum",columNum)
#     for i in range(0,rowNum):
#         print(rowNum,"=",i)
#         for j in range(0,columNum-1):
#             sheet.cell(row = i+1,column= j+1).value=tableValue[i][j+1]
#     print("wait")
#     wb.save(path)
#     print("写入数据成功！")
#
#
# def read07excel(path):
#     wb2=load_workbook(path)
#     #print(wb2.get_sheet_names())
#     sheet=wb2.get_sheet_by_name("Sheet1")
#     row=sheet.get_highest_row()
#     col=sheet.get_highest_column()
#     print("列数: ",sheet.get_highest_column())
#     print("行数: ",sheet.get_highest_row())
#
#     tableValue=list()
#     for i  in range(0,row):
#         print("rowNo",i,"==",end="")
#         rowValue=list()
#         for j in range(0,col):
#             cellValue=sheet.rows[i][j].value
#             if(cellValue==None and j<5 and i>0):
#                 cellValue=tableValue[i-1][j]
#             elif(cellValue=="同上"):
#                 cellValue=tableValue[i-1][j]
#             rowValue.append(cellValue)
#             # if(cellValue==None):
#             #     print("列",j,"\t\t",end="")
#         print(len(rowValue),rowValue)
#         tableValue.append(rowValue)
#     return tableValue
#     #print(ws.rows[0][0].value)
#     #print(ws.rows[1][0].value)
#     #print(ws.rows[0][1].value)
#
# #def writeexcel03(path):
# #
# #    wb=xlwt3.Workbook()
# #    sheet=wb.add_sheet("xlwt3数据测试表")
# #    value = [["名称", "hadoop编程实战", "hbase编程实战", "lucene编程实战"], ["价格", "52.3", "45", "36"], ["出版社", "机械工业出版社", "人民邮电出版社", "华夏人民出版社"], ["中文版式", "中", "英", "英"]]
# #    for i in range(0,4):
# #        for j in range(0,len(value[i])):
# #            sheet.write(i,j,value[i][j])
# #    wb.save(path)
# #    print("写入数据成功！")
#
#
# #excelpath=r"D://名称.xlsx"
# #writepath=r"D://书籍明细07.xlsx"
# #writeexcel03(writepath)
# #writeexcel07(writepath)
#
# #read07path="D://名称.xlsx";
#
#
# # read03path=r"D:\workingDirectory\信访分析\金融局信访\12345数据\转办整理后数据\12345整理数据.xls";
# read07path=r"D:\workingDirectory\信访分析\金融局信访\12345数据\转办整理后数据\合并.xlsx";
# # read07path=r"D:\workingDirectory\信访分析\金融局信访\12345数据\转办整理后数据\test.xlsx";
# write07path=r"D:\workingDirectory\信访分析\金融局信访\12345数据\转办整理后数据\合并fix.xlsx";
# tableValue=read07excel(read07path)
# writeexcel07(write07path,tableValue)
# # writeexcel07(write07path,tableValue)
# #read07excel(read03path)
# #showexcel(excelpath);
# # showexcel(read03path);