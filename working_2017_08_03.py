#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 23:17:15 2017

@author: Chang-Eop
"""


import os
os.chdir(r'C:\Users\JH\Desktop\TextMining\PainNetwork')



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import codecs
import csv
import re
import time



data = pd.read_excel('NIFSTD.xlsx')
data_1 = data.ix[:,['Preferred Label','Synonyms', 'has_obo_namespace']]

#data_1.ix[data_1.ix[:,0] == 'amygdala',:]

data_2 = data_1[data_1.ix[:,2] == 'uberon']
data_2.index = range(data_2.shape[0])



#NIFSTD의 동의어 정리
terms = [] #nested list 형식으로 동의어들 묶어넣을것
for i in range(data_2.shape[0]):
    term_prefer = data_2.ix[i,0]
    try: #동의어 없는 경우(nan) 대비.
        term_synonym = data_2.ix[i,1].split("|")
    except AttributeError:
        term_synonym = []
    
    temp = [term_prefer]
    temp.extend(term_synonym) #제일 앞이 prefer, 이어서 동의어들.
    terms.append(temp)


        
    
        
    
candi_terms = pd.read_excel('ROI_Brede.xlsx')#'170525_RPO_Brede(except_Broadmann_Economo_Monkey)_1107.xlsx')
#오지홍 정리 


#candidate terms 모으기    (nan 빼고, 그냥 다 리스트로 만든것)
candi_total = []
for i in range(candi_terms.shape[0]):
    for j in range(candi_terms.shape[1]):
        if type(candi_terms.ix[i,j]) == str: 
            candi_total.append([candi_terms.ix[i,j]])
        else:
            pass
        
        
# candidate terms 중에 NIFSTD 동의어 목록에서 찾아지는 수
count = 0        
for candi in candi_total:
    
    for t in range(len(terms)):
        if candi[0].lower() in terms[t]:
            count += 1
            
            

#text_total = pd.read_excel('pain_papers.txt')

f = codecs.open('pain_papers.txt','r',"utf-8")
texts = f.readlines() #초록 하나가 리스트의 요소 하나로 들어감.

#전체 소문자로 변환
for i,line in enumerate(texts):
    texts[i] = line.lower()



#candidate terms의 동의어들로 쿼리 구성하고, 문서 전체에서의 등장 빈도 카운트
freq_of_candis = {}

n = 0
for candi in candi_total:
    counts = 0
    n += 1
    
    #단어 앞뒤 공백 제거
    candi = [candi[0].strip()]
    #동의어 포함한 쿼리 구성
    for t in range(len(terms)):
        if candi[0].lower() in terms[t]: #동의어 목록에 있으면 
            query = terms[t] #동의어 리스트를 쿼리로
            break #찾아으면 다음 루프로
        else: #동의어 목록에 없으면 
            query = [candi[0].lower()] #그냥 그 자체만 리스트 형태로
    
    #전체 텍스트에서 쿼리(동의어 포함) 등장 빈도 카운트(문서당 최대 1회 카운트)
    for qs in query:
        for i, text in enumerate(texts):
            if qs in text: #해당 용어 발견했을때
                
                pattern_0 = re.compile(qs) #용어 존재    
                pattern_1 = re.compile('[a-z]'+qs)#용어 앞에 다른 문자 연결되어있거나
                pattern_2 = re.compile(qs+'[a-z]')#용어 뒤에 다른 문자 연결된 경우라면
                
                iters = pattern_0.finditer(text) #용어 존재하는 경우를  iterable 객체로 반환
                
                for it in iters: #용어 존재 경우 하나씩 돌아가면서
                    start = it.span()[0] #텍스트 상에서의 시작위치
                    end = it.span()[1] #텍스트 상에서의 용어 종료 위치
                    test_phrase = text[start-1:end+1] #용어 앞뒤로 한칸 추가하여 따냄
                    if not (pattern_1.search(test_phrase)) and not (pattern_2.search(test_phrase)):
                        # 앞뒤로 알파벳 있지 않은 경우만 카운트
                        counts += 1
                        break #하나라도 찾아서 카운트했으면 다음 텍스트로 넘어감
       
                
    freq_of_candis[candi[0]] = counts
    print(counts, ", ", n, " of ", len(candi_total))
        

#결과 csv 포맷으로 저장
f2 = open('FreqCandis_ROI_Brede.csv', 'w')
w = csv.writer(f2)

for r_i, row in enumerate(freq_of_candis.items()):
    if r_i == 0:
        w.writerow(('Node', 'Freq')) #pandas 대비하여 컬럼 명 추가
    else:
        w.writerow(row)
f2.close()



         

        
#이하 작업중 시험 코드 
nodes = pd.read_csv('FreqCandis_ROI_Brede.csv')   
nodes = nodes.sort_index(by = 'Freq', ascending=False)

nodes = nodes[nodes.Node != 'Trochlear nucleus'] #동의어 4 때문에 오류
nodes = nodes[nodes.Node != 'Motor cortex'] # motor area 동의어
nodes = nodes[nodes.Node != 'Primary motor cortex'] # motor area 동의어
nodes = nodes[nodes.Node != 'Hippocampal formation']  # hippocampus 동의어
nodes = nodes[nodes.Node != 'Cerebrospinal fluid'] # 적절치 않다고 판단
nodes = nodes[nodes.Node != 'Archicortex'] #동의어에 hippocampus가 있어 빈도 높게 나오는것으로 보임.
nodes = nodes[nodes.Node != 'Lateral occipito-temporal gyrus'] #Fusiform gyrus 동의어
nodes = nodes[nodes.Node != 'Central nuclear group'] #Central amygdaloid nucleus 동의어
nodes = nodes[nodes.Node != 'Medial frontal gyrus'] #Middle frontal gyrus 동의어
#nodes = nodes[nodes.Node != 'Reticular formation'] #Medullary reticular formation 인데, reticular formation도 preferred에 있어서 고려할 필요

             
             
nodes = nodes[nodes.ix[:,1] > 20] # 일정 빈도 이상만.

             
nodes.to_csv('pain_FreqCandis_ROI_Brede_nodes.csv')   #nodes 를 csv 파일로 저장      
     
#list(data_2[data_2.ix[:,0] == 'fusiform gyrus']['Synonyms']) # 동의어 겹치는애들 확인해보는 코드            
            
#시간은 오래 걸리지만, 처리 안해도 최종 결과에 영향 업는것으로 (일단) 판단되어 비활성화.
## texts에서 \n 제거 (시간 오래 걸림)
#ns = texts.count('\n')
#for i in range(ns): #'\n' in texts:
#    texts.remove('\n')
#    print(i+1, " of ", ns)
# 
## 이유 알수 없지만 '.\n'이 리스트에 섞여 있음.    
#for i in range(texts.count('.\n')):
#    texts.remove('.\n')
    
    



#texts를 text_lists로 정리.
#texts는 abstract가 쪼개어져 있는 경우가 있으므로 이를 다시 정리함. 
#요소 하나는 하나의 paper 정보 포함하도록 (year, abstract)가 되도록.
#시간 오래 걸림!!!!!!!

text_list = [] #엘리먼트 개별  (임시적으로 assign)
text_lists = [] 
for i, text in enumerate(texts):
    print(i+1, len(texts))
    try:
        year = int(text)
        text_lists.append(text_list)
        text_list = [year]
    except ValueError:
        text_list.append(text)
        
text_lists.append(text_list) #맨 마지막은 따로 추가해야 함
text_lists=text.lists[1:] #맨 앞에 '\n' 반복되는 꾸러미 하나 있어서 제거
#text_lists.remove([]) #162-171 안 돌리면 얘 필요 없음 .제일 첫 엘리먼트가 공리스트라 제거.
             
                 
                 
                 
t_start = time.time()  
#occur_matrix 생성
#abstract별로 np.array (row: abstracts, col: year, freq. of nodes) 생성
occur_matrix = np.zeros((len(text_lists), len(nodes)+1))
for i, year_abstract in enumerate(text_lists):
    print(i+1, " of ", len(text_lists))
    occur_matrix[i,0] = year_abstract[0] #year
    
    #            
    for j, node in enumerate(nodes.Node):
        node = node.lower()
        
        #노도의 동의어 목록으로 쿼리 구성
        for t in range(len(terms)):
            if node in terms[t]: #동의어 목록에 있으면 
                query = terms[t] #동의어 리스트를 쿼리로
                break #찾아으면 다음 루프로
            else: #동의어 목록에 없으면 
                query = [node] #그냥 그 자체만 리스트 형태로
                
                
                
                
                
        #초록에 쿼리(동의어 포함) 등장 여부 결정(카운트 아닌 여부(1/0))
        val = [] #동의어 각각의 등장 여부를 1/0 리스트로 만들것.
        for qs in query:
            
            if qs in year_abstract[1]: #해당 용어 발견했을때
                
                pattern_0 = re.compile(qs) #용어 존재    
                pattern_1 = re.compile('[a-z]'+qs)#용어 앞에 다른 문자 연결되어있거나
                pattern_2 = re.compile(qs+'[a-z]')#용어 뒤에 다른 문자 연결된 경우라면
                
                iters = pattern_0.finditer(year_abstract[1]) #용어 존재하는 경우를  iterable 객체로 반환
                
                for it in iters: #용어 존재 경우 하나씩 돌아가면서
                    start = it.span()[0] #텍스트 상에서의 시작위치
                    end = it.span()[1] #텍스트 상에서의 용어 종료 위치
                    test_phrase = year_abstract[1][start-1:end+1] #용어 앞뒤로 한칸 추가하여 따냄
                    if not (pattern_1.search(test_phrase)) and not (pattern_2.search(test_phrase)):
                        # 앞뒤로 알파벳 있지 않은 경우만 카운트
                        val.append(1)
                        break #하나라도 찾아서 카운트했으면 다음 텍스트로 넘어감
        occur_matrix[i,j+1] = int(1 in val)                
                
    
#plt.matshow(occur_matrix[:500,1:])

t_end = time.time()          
        
print(t_end - t_start)         
                    
                

#>>> (t_end - t_start)/3600   
#9.949767874413066  
 
#year에 따라 sorting                
occur_matrix  = occur_matrix[np.argsort(occur_matrix[:,0])]                 
          
#연도별 총 초록 수
y_min = int(min(occur_matrix[:,0]))
y_max = int(max(occur_matrix[:,0]))
y_hist = np.zeros((y_max - y_min +1, 1))
for y_i, y in enumerate(range(y_min, y_max+1)):
    y_hist[y_i] = sum(occur_matrix[:,0] == y)

#첫번째 칼럼에 연도 배치    
y_hist = np.array([np.arange(y_min, y_max+1), y_hist[:,0]]).T

plt.figure()
plt.plot(y_hist[:,0], y_hist[:,1])

#y_hist_2: 최초-1975년까지 초록수 합하여 1975년으로 할당
y_hist_2 = y_hist[y_hist[:,0] == 1975]
y_hist_2[:,1] = np.sum(y_hist[y_hist[:,0] < 1976,1])
y_hist_2 = np.concatenate((y_hist_2,y_hist[y_hist[:,0] > 1975]))


#occur_matrix_2: 최초 - 1975년 초록까지 모두 1975년으로 처리
occur_matrix_2 = occur_matrix[:]
occur_matrix_2[occur_matrix_2[:,0] < 1976,0] = 1975

plt.figure()
plt.plot(y_hist_2[:,0], y_hist_2[:,1])


#
rel_freq = np.zeros((len(y_hist_2), len(nodes)))
for y_i, y in enumerate(range(int(y_hist_2[0,0]), int(y_hist_2[-1,0]))):
    print(y_i, y)
    freq_sum_year = occur_matrix[occur_matrix[:,0] == y].sum(axis = 0)/y_hist_2[y_i,1]
    freq_sum_year = freq_sum_year[1:]
    rel_freq[y_i,:] = freq_sum_year


plt.figure()
plt.matshow(rel_freq[:,:])
plt.yticks(range(len(y_hist_2)), y_hist_2[:,0].astype(int), Fontsize = 6)
plt.xticks(range(len(nodes.Node)),nodes.Node, rotation = 90, Fontsize = 6)


    
#occur_matrix_3: 노드 포함 안하는 초록 제거, 연도 제거
occur_matrix_3 = occur_matrix_2[np.sum(occur_matrix_2[:,1:], axis = 1) > 0,1:]
    
#전체 초록에 대한 co-occurence matrix  만들기
corr_total = np.dot(occur_matrix_3.T, occur_matrix_3)

plt.figure()
plt.matshow(corr_total)
plt.yticks(range(len(nodes.Node)),nodes.Node, Fontsize = 6)
plt.xticks(range(len(nodes.Node)),nodes.Node, rotation = 90, Fontsize = 6)

            
#연도별로 co-occurence matrix 구성(row, col: node 수, depth: 연도 수)
corr_years = np.zeros((corr_total.shape[0], corr_total.shape[1],len(y_hist_2)))
for y_i, y in enumerate(y_hist_2[:,0]):
    occur_matrix_4 = occur_matrix_2[occur_matrix_2[:,0] == y,1:]
    corr_years[:,:,y_i] = np.dot(occur_matrix_4.T, occur_matrix_4)
    
    
#관심 연도(year of interest) 설정하여 co-occurence matrix 그리기

#yoi = 2013
#yoi_i = np.where(y_hist_2[:,0] == yoi)[0][0]

#plt.figure()
#plt.matshow(corr_years[:,:,yoi_i])
#plt.title(yoi)
#plt.yticks(range(len(nodes.Node)),nodes.Node, Fontsize = 6)
#plt.xticks(range(len(nodes.Node)),nodes.Node, rotation = 90, Fontsize = 6)      
 
     
#musun 도움. co-occur matrix 1975-2016 연속으로 뽑는 코드      

for i in range(len(y_hist_2)-1):
    yoi = i
    plt.figure(figsize = (50,50))
    plt.set_cmap('Reds')
    plt.imshow(corr_years[:,:,i])
    plt.colorbar(plt.imshow(corr_years[:,:,i],Fontsize=20)
    plt.title(str(yoi+1975), Fontsize=100)
    plt.yticks(range(len(nodes.Node)),nodes.Node, Fontsize = 20)
    plt.xticks(range(len(nodes.Node)),nodes.Node, rotation = 90, Fontsize = 20)
    plt.savefig(str(yoi+1975) +'.png')

        
        