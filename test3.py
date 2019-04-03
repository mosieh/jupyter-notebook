#!/usr/bin/env python
# coding: utf-8

# In[1]:


import math


# In[2]:


math.sqrt (36)


# In[3]:


math.pi


# In[4]:


mt.sqrt(25)


# In[5]:


import math as mt


# In[6]:


mt.sqrt(25)


# In[7]:


import numpy


# In[8]:


import stats from scipy


# In[9]:


import scipy.stats


# In[10]:


from numpy import random


# In[11]:


import numpy as np


# In[12]:


#uniform random numbers in [0,1]


# In[13]:


dataOne=random.rand(5,5)


# In[14]:


np.mean(dataOne)


# In[15]:


array(dataOne)


# In[16]:


dataOne


# In[17]:


from scipy.stats import itemfreq


# In[18]:


print y.shape, itemfreq(y)


# In[19]:


np.mean(dataOne)


# In[20]:


np.sum(dataOne)


# In[21]:


np.average(dataOne)


# In[22]:


np.min(dataOne)


# In[23]:


np.max(dataOne)


# In[24]:


from scipy import stats


# In[25]:


import scipy as sp


# In[26]:


import matplotlib as mpl
from matplotlib import pyplot as plt


# In[27]:


sp.std(dataOne)


# In[28]:


sp.var(dataOne)


# In[29]:


sp.skew(dataOne)


# In[30]:


sp.min(dataOne)


# In[31]:


print('I Love Data Science')


# In[32]:


variableName=25


# In[33]:


variableName1='I Love Data Science'


# In[34]:


print(variableName1)


# In[35]:


variableName=25


# In[36]:


type(variableName)


# In[37]:


variableName=25.0


# In[38]:


type(variableName)


# In[39]:


varOne=25


# In[40]:


varTwo=25.0


# In[41]:


varThree=varOne + varTwo


# In[42]:


print(varThree)


# In[43]:


varOne=25


# In[44]:


varTwo='Hello'


# In[45]:


varThree = varOne + varTwo


# In[46]:


c=ord(varTWo)


# In[47]:


c=ord(varTwo)


# In[48]:


c=int(varTwo)


# In[49]:


c=hex(varTwo)


# In[50]:


listOne=[1,2,3,4]


# In[51]:


print(listOne[1:3])


# In[52]:


tel={'jack':4098,'sape':4139}


# In[53]:


tel['jack']


# 

# In[54]:


dog={'MJ':6,'belle':15,'ella':4,'minnie':3, 'penny':1}


# In[55]:


np.mean(dog)


# In[56]:


print(dog)


# In[57]:


ord("A")


# In[58]:


myInt=int("123")


# In[59]:


myStr=str(1234.56)


# In[60]:


import datetime


# In[61]:


datetime.datetime.now()


# In[62]:


str(datetime.dateeime.now().date)


# In[63]:


str(datetime.datetime.now().date)


# In[64]:


str(datetime.datetime.now().date())


# In[65]:


def SayHello():
    print('Hello There!')


# In[66]:


SayHello()


# In[67]:


def DoSum(Value1, Value2):
    return Value1 + Value2


# In[68]:


DoSum


# In[69]:


DoSum(1,2)


# In[70]:


def DisplaySum(Value1, Value 2):
    print (str(Value1)+'+'+str(Value2) + '='+ 
          str((Value1 + Value2)))


# In[71]:


def DisplaySym(Value1, Value2):
    print(str(Value1)+'+'+ str(Value2)+'='+ 
          str((Value1 + Value 2)))


# In[72]:


def DisplaySym(Value1, Value2):
    print(str(Value1)+'+'+ str(Value2)+'='+ 
          str((Value1 + Value2)))


# In[73]:


DisplaySum(2,3)


# In[74]:


def DisplaySum(Value1, Value2):
    print(str(Value1)+'+'+str(Value2)+'='+
         str((Value1 + Value2)))


# In[75]:


DisplaySum(2,3)


# In[76]:


DisplaySum(Value2=3, Value1=2)


# In[77]:


def SayHello(Greeting = "No Value Supplied"):
    print(Greeting)


# In[78]:


SayHello()


# In[79]:


SayHello("Howdy!")


# In[80]:


def DisplayMulti (ArgCount=0,*VarArgs):
    print('You passed'+str(ArgCount)+'arguments.',
         Var Args)


# In[81]:


def DisplayMulti(ArgCount=0,*VarArgs):
    print('You passed'+ str(ArgCount)+'arguments.',
         VarArgs)


# In[82]:


DisplayMulti()


# In[83]:


DiplayMulti(3,'Hello',1,True)


# In[84]:


def DisplayMulti((ArgCount=0,*VarArgs):
    print('You passed'+ str(ArgCount)+'arguments.',
         VarArgs))


# In[85]:


def DisplayMulti(ArgCount=0, *VarArgs):print('You passed'+str(ArgCount)+'arguments.',VarArgs)


# In[86]:


DisplayMulti(3,'Hello',1,True)


# In[87]:


def TestValue(Value):
    if Value == 5:
        print('Value equals 5!')
        elif Value == 6:
            print('Value equals 6!')
            else:
                print('Value is something else.')
                print('It equals'+str(Value))


# In[88]:


def TestValue(Value):
    if Value == 5:
        print('Value equals 5!')
        elif Value == 6:
            print('Value equals 6!')
            else:
                print('Value is something else.')
                print('It equals' + str(Value))
                


# In[89]:


def TestValue(Value):
    if Value == 5:
        print('Value equals 5!')
        else if Value == 6:
            print('Value equals 6!')
            else:
                print('Value is something else.')
                print('It equals' + str(Value))
                


# In[90]:


def TestValue(Value):
    if Value == 5:
        print('Value equals 5!')
    elif Value == 6:
        print('Value equals 6!')
    else:
        print('Value is something else.')
        print('It equals'+ str(Value))
        


# In[91]:


TestValue(1)


# In[92]:


TestValue(5)


# In[93]:


TestValue(6)


# In[94]:


def SecretNumber():
    One = int(input("Type a number between 1 and 10:"))
    Two = int(input("Type a number between 1 and 10:"))
    
    if (One >= 1) and (One <= 10):
        if (Two >= 1) and (Two <= 10):
            print('Your secret number: '+ str(One * Two))
        else:
            print("Incorrect second value!")
    else:
        print("Incorrect first value!")


# In[95]:


SecretNumber()


# In[96]:


def DisplayMulti(*VarArgs):
    for Arg in VarArgs:
        if Arg.upper()=='CONT':
            continue
            print('Continue Argument: '+ Arg)
        elif Arg.upper() == 'BREAK':
            break
            print('Break Argument: '+ Arg)
        print('Good Argument: '+ Arg)


# In[97]:


DisplayMulti('Hello','Goodbye','First','Last')


# In[98]:


DisplayMulti('Hello','Cont','Goodbye','Break','Last')


# In[99]:


def SecretNumber():
    GotIt = False
    whiel GotIt == False:
        One = int(input("Type a number between 1 and 10:"))
        Two = int(input("Type a number between 1 and 10:"))
        
        if (One>= 1) and (One <= 10):
            if (Two >= 1) and (Two <= 10):
                print ('Secret number is:' + str (One*Two))
                GotIt = True
                continue
            else: 
                print("Incorrect second value!")
        else:
            print("Incorrect first value!")
        print("Try again!")


# In[100]:


def SecretNumber():
    GotIt = False
    while GotIt == False:
        One = int(input("Type a number between 1 and 10:"))
        Two = int(input("Type a number between 1 and 10:"))
        
        if (One>= 1) and (One <= 10):
            if (Two >= 1) and (Two <= 10):
                print ('Secret number is:' + str (One*Two))
                GotIt = True
                continue
            else: 
                print("Incorrect second value!")
        else:
            print("Incorrect first value!")
        print("Try again!")


# In[101]:


SecretNumber()


# In[102]:


from sets import Set
SetA = Set (['Red','Blue','Green','Black'])
SetB = Set(['Black','Green','Yellow','Orange'])
SetX = SetA.union(SetB)
SetY = Seta.intersection(SetB)
SetZ = SetA.difference(SetB)


# In[103]:


from sets import Set


# In[104]:


import set


# In[105]:


import sets


# In[106]:


import Set


# In[107]:


import Sets


# In[108]:


print('{0}\n{1}\n{2}'.format(SetX, SetY, SetZ))


# In[109]:


SetA = Set (['Red','Blue','Green','Black'])
SetB = Set(['Black','Green','Yellow','Orange'])
SetX = SetA.union(SetB)
SetY = Seta.intersection(SetB)
SetZ = SetA.difference(SetB)


# In[110]:


SetA = Set(['Red','Blue','Green','Black'])
SetB = Set(['Black','Green','Yellow','Orange'])
SetX = SetA.union(SetB)
SetY = Seta.intersection(SetB)
SetZ = SetA.difference(SetB)


# In[111]:


print('{0}\n{1}\n{2}'.format(SetX, SetY, SetZ))


# In[112]:


SetA = Set(['Red','Blue','Green','Black'])
SetB = Set(['Black','Green','Yellow','Orange'])
SetX = SetA.union(SetB)
SetY = SetA.intersection(SetB)
SetZ = SetA.difference(SetB)


# In[113]:


ListA = [0,1,2,3]
ListB = [4,5,6,7]
ListA.extend(ListB)
ListA


# In[114]:


ListA.append(5)


# In[115]:


ListA


# In[116]:


ListA.remove(5)


# In[117]:


ListA


# In[118]:


ListA = [0,1,2,3]
ListB = [4,5,6,7]
ListA.extend(ListB)
ListA


# In[119]:


ListA


# In[120]:


ListA.append(-5)


# In[121]:


ListA


# In[122]:


ListA.remove(-5)


# In[123]:


ListA


# In[124]:


ListX= ListA+ListB


# In[125]:


ListX


# In[ ]:





# In[126]:


MyTuple = (1,2,3, (4,5,6(7,8,9)))


# In[127]:


MyTuple = (1,2,3 (4,5,6 (7,8,9)))
for Value 1 in MyTuple:
    if type(Value1)== int:
        print Value1
    else:
        for Value2 in Value1:
            if type(Value2) == int:
                print "\t", Value2
            else:
                for Value3 in Value2:
                    print "\t\t", Value3


# In[128]:


MyTuple = (1,2,3 (4,5,6 (7,8,9)))
for Value1 in MyTuple:
    if type(Value1)== int:
        print Value1
    else:
        for Value2 in Value1:
            if type(Value2) == int:
                print "\t", Value2
            else:
                for Value3 in Value2:
                    print "\t\t", Value3
                    


# In[129]:


MyTuple = (1,2,3 (4,5,6 (7,8,9)))
for Value1 in MyTuple:
    if type(Value1)== int:
        print(Value1)
    else:
        for Value2 in Value1:
            if type(Value2) == int:
                print "\t", Value2
            else:
                for Value3 in Value2:
                    print "\t\t", Value3


# In[130]:


MyTuple = (1,2,3 (4,5,6 (7,8,9)))
for Value1 in MyTuple:
    if type(Value1)== int:
        print(Value1)
    else:
        for Value2 in Value1:
            if type(Value2) == int:
                print ("\t", Value2)
            else:
                for Value3 in Value2:
                    print ("\t\t", Value3)


# In[131]:


ListA = ['Orange','Yellow','Green','Brown']
ListB = [1,2,3,4]


# In[132]:


ListA[1]


# In[133]:


ListB[1:3]


# In[134]:


for Value in ListB[1:3]:
    print Value


# In[135]:


for Value in ListB[1:3]
print (Value)


# In[136]:


for Value in ListB[1:3]
print (Value)


# In[137]:


for Value in ListB[1:3],
print (Value)


# In[138]:


for Value1, Value2 in zip(ListA, ListB):
    print (Value1, '\t', Value2)


# In[139]:


MyDict ={'Orange':1, 'Blue':2,'Pink':3}


# In[140]:


MyDict['Pink']


# In[141]:


SetA = set('Red','Blue','Green','Black')
SetB = set('Black','Green','Yellow','Orange')
SetX = SetA.union(SetB)
SetY= SetA.intersection(SetB)
SetZ = SetA.difference(SetB)


# In[142]:


print('{0}\n{1}\n{2}'.format(SetX, SetY, SetZ))


# In[143]:


SetA = set('Red','Blue','Green','Black')


# In[144]:


set = (a)


# In[145]:


with open ("Colors.txt",'r')as open_file:
    print 'Colors.txt content:\n' + open_file.read()


# In[146]:


with open ("Colors.txt",'rb')as open_file:
    print 'Colors.txt content:\n' + open_file.read()


# In[147]:


with open ("Colors.txt",'rb') open_file:
    for observation in open_file:
        print 'Reading Data:'+observation


# In[148]:


SetA = {'Red','Blue','Green','Black'}
SetB = {'Black','Green','Yellow','Orange'}
SetX= A.union(SetB)
SetY = A.intersection(SetB)
SetZ = A.difference(SetB)


# In[149]:


SetA = {'Red','Blue','Green','Black'}
SetB = {'Black','Green','Yellow','Orange'}
SetX = A.union(SetB)
SetY = A.intersection(SetB)
SetZ = A.difference(SetB)


# In[150]:


SetA = {'Red','Blue','Green','Black'}
SetB = {'Black','Green','Yellow','Orange'}
SetX = SetA.union(SetB)
SetY = SetA.intersection(SetB)
SetZ = SetA.difference(SetB)


# In[151]:


print('{0}\n{1}\n{2}'.format(SetX, SetY, SetZ))


# In[152]:


with open("Colors.txt", 'rb') as open_file:
    print('Colors.txt content:\n'+ open_file.read())


# In[153]:


with open("Colors.txt", 'rb') as open_file:
    print('Colors.txt content:\n'+ open_file.read())


# In[154]:


http://localhost:8888/edit/C5T1/Colors.txt#
        


# In[155]:


import CSV
With open(‘Colors.txt’, ‘rb’) as f:
reader = csv.reader(f)
for row in reader:
print row


# In[156]:


import CSV
open ('Colors.txt','rb')as f:
reader=csv.reader(f)
for row in reader:
print (row)
    


# In[157]:


import csv


# In[158]:


with open ('Colors.txt','r') as csvfile:
    print 'Colors.txt content:\n' + csvfile.read()


# In[159]:


import csv
with open('Colors.txt', newline='') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)


# In[160]:


Color.txt


# In[161]:


with open('Colors.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')


# In[162]:


with open('/Users/mosiehackett/C5T1/Colors.txt') as open_file:
          print '/Users/mosiehackett/C5T1/Colors.txt content' + open_file.read()


# In[163]:


with open('/Users/mosiehackett/C5T1/Colors.txt') as open_file:
          print '/Users/mosiehackett/C5T1/Colors.txt content' + open_file.read()


# In[164]:


with open('/Users/mosiehackett/C5T1/Colors.txt') as open_file:
          print '/Users/mosiehackett/C5T1/Colors.txt' + open_file.read()


# In[165]:


import sys


# In[1]:


with open("Colors.txt",'r') as open_file:
    print('Colors.txt content: \n' + open_file.read())


# In[2]:


with open("Colors.txt",'r') as open_file:
    print('Colors.txt content: \n' + open_file.read())


# In[3]:


with open("Colors.txt", 'r') as open_file:
    for observation in open_file:
        print 'Reading Data:'+ observation


# In[4]:


with open("Colors.txt",'r') as open_file:
    for observation in open_file:
        print ('Reading Data:'+ observation)


# In[5]:


n=2
with open ("Colors.txt",'r')as open_file:
    for j, observation in enumerate (open_file):
        if j % n==0:
            print('Reading Line:'+ str(j)+
                 'Content:' + observation)


# In[6]:


from random import random
sample_size = 0.25
with open("Colors.txt",'r')as open_file:
    for j,observation in enumerate(open_file):
        if random()<=sample_size:
            print('Reading Line:'+ str(j)+
                 'Content:'+ observation)


# In[7]:


import pandas as pd
color_table=pd.io.parsers.read_table("Colors.txt")
print color_table


# In[8]:


import pandas as pd
color_table=pd.io.parsers.read_table("Colors.txt")
print (color_table)


# In[9]:


import pandas as pd
titanic=pd.io.parsers.read_csv("Titanic.csv")
X = titanic[['age']]
print (X)


# In[10]:


import pandas as pd
xls=pd.ExcelFile("Values.xls")
trig_values = xls.parse('Sheet1',index_col=None,
                       na_values=['NA'])


# In[11]:


import pandas as pd
xls=pd.ExcelFile("Values.xls")
trig_values = xls.parse('Sheet1',index_col=None,
                       na_values=['NA'])
print (trig_values)


# In[12]:


from skimage.io import imread
from skimage.transform import resize
from matplotlib impor pyplot as plt
import matplotlib.cm as cm

example_file=("http://upload.wikimedia.org/"+
             "wikipedia/commons/7/7d/Dog_face.png")
image=imread(example_file, as_grey=True)
plt.imshow(image,cmap=cm.gray)
plt.show()


# In[13]:


from skimage.io import imread
from skimage.transform import resize
from matplotlib import pyplot as plt
import matplotlib.cm as cm

example_file=("http://upload.wikimedia.org/"+
             "wikipedia/commons/7/7d/Dog_face.png")
image=imread(example_file, as_grey=True)
plt.imshow(image,cmap=cm.gray)
plt.show()


# In[14]:


print("data type: %s, shape:%s"%
     (type(image), image.shape))


# In[15]:


image2=image[5:70,0:70]
plt.imshow(image2,cmap=cm.gray)
plt.show()


# In[16]:


image3 = resize(image2,(30,30), mode='nearest')
plt.imshow(image3,cmap=cm.gray)
print("data type:%s, shape:%s"% 
     (type(image3),image3.shape))


# In[17]:


image_row=image3.flatten()
print("data type: %s, shape:%s" %
     (type(image_row),image_row.shape))


# In[18]:


from lxml import objectify
import pandas as pd

xml= objectify.parse(open('XMLData.xml'))
root= xml.getroot()

df=pd.DataFrame(columns=('Number','String','Boolean'))

for i in range(0,4):
    obj=root.getchildren()[i].getchildren()


# In[19]:


from lxml import objectify
import pandas as pd

xml= objectify.parse(open('XMLData.xml'))
root= xml.getroot()

df=pd.DataFrame(columns=('Number','String','Boolean'))

for i in range(0,4):
    obj=root.getchildren()[i].getchildren()
    row= dict(zip(['Number','String','Boolean'],
                 [obj[0].text,obj[1].text,
                 obj[2].text]))
    row_s=pd.Series(row)
    row_s.name=1
    df=df.append(row_s)
    
print(df)    


# In[20]:


with open("iri.csv",'r') as open_file:
    print('iris.csv content: \n' + open_file.read())


# In[21]:


with open("iris.csv",'r') as open_file:
    print('iris.csv content: \n' + open_file.read())


# In[ ]:




