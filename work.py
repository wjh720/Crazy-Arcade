from env import Maze
from tkinter import *
from baseline import baseline
import time
import cv2

'''
Map =\
['##########',
 '#^#      #',
 '# #oo### #',
 '# #    # #',
 '# # o#^# #',
 '#     ## #',
 '#  oooo  #',
 '#  o^*o  #',
 '#  oooo  #',
 '#0      1#',
 '##########',
]
'''
Map=\
['#o#*#o#*#',
 'o1o  *o^o',
 '# # # #o#',
 'o #####^o',
 '#*#####o#',
 'oo^oo^  o',
 '#oo^#oo0#'
]

human_play=False
have_render=True

env = Maze(Map)
if(have_render):
    Image = env.render()

act=[0,0]
score=[0,0]

def ch(x):
    if(x=='l'):return 0
    if(x=='r'):return 1
    if(x=='u'):return 2
    if(x=='d'):return 3
    if(x=='b'):return 4
    if(x=='s'):return 5
def read_action(event):
    if event.keysym == 'Left':
        act[0]='l'
    if event.keysym == 'Right':
        act[0]='r'
    if event.keysym == 'Up':
        act[0]='u'
    if event.keysym == 'Down':
        act[0]='d'
    if event.keysym == 'space':
        act[0]='b'
    if event.keysym == 'Control_L':
        act[0]='s'
    
    act[1]=baseline.choose_action(env,1)

    if(act[0]!=0 and act[1]!=0):
        res=env.step({(0,ch(act[0])),(1,ch(act[1]))})
        
        cv2.imshow("Image", res[0])
        cv2.waitKey()
        cv2.destroyAllWindows()
        
        env.render()
        cnt=0
        score[0]+=res[0][0]
        score[1]+=res[0][1]
        #print('score:',score[0],score[1])
        act[0]=0
        act[1]=0
    else:
        if(act[0]==0 and act[1]==0):
            return
        if(act[0]==0):
            print('Wait for P0')
        else: print('Wait for P1')
    
    print('你按下了: ' + event.keysym)


if human_play:
    Image.bind('<Key>', read_action)
else:
    while(True):
        env.reset()
        stepcnt=0
        start=time.time()
        score=[0,0]
        if(have_render):
            env.render()   # render
        while(True):
            stepcnt+=1
            act[0]=baseline.choose_action(env,0)
            act[1]=baseline.choose_action(env,1)
            res=env.step({(0,ch(act[0])),(1,ch(act[1]))})
            
            cv2.imshow("Image", res[0])
            cv2.waitKey()
            cv2.destroyAllWindows()
            
            if(have_render):
                env.render()   # render
            cnt=0
            score[0]+=res[1][0]
            score[1]+=res[1][1]
            print('score:',score[0],score[1])
            if res[2]:
                break
        if(have_render==True):
            time.sleep(1)
        print("time= %lf %lf" %(time.time()-start, stepcnt))
if(have_render):
    Image.mainloop()
