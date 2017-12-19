
import numpy as np
import time
import sys
import state_creator
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
from PIL import Image, ImageTk

#import atari_py

UNIT = 40   # pixels
MAZE_H = -1  # grid height, given by initializing
MAZE_W = -1  # grid width, given by initializing


'''
decide
'''
BOOM_R = 1   # boom explosion rage, from x to x+BOOM_R
BOOM_T = 5   # time until explosion
BOOM_N = 1   # initial number of boom
GAME_ROUND = 50 #maximum game ronud
REWARD_BOX = 10 #reward of box
REWARD_ITEM = 5  #reward of item
REWARD_KILL =100 #reawrd of kill opponent
PUNISH = 50 #punishment
REWARD_BOMB=200
PUNISH_PER_ROUND=1

class player:
    def __init__(self):
        self.players=None
        self.xy=-1
        self.max_boom_num=BOOM_N
        self.boom_num=0
        self.boom_r=BOOM_R
        self.fig=None

def resize(pil_image):  
    w,h = pil_image.size  
    f1 = 1.0*39/w
    f2 = 1.0*39/h
    factor = min([f1, f2])  
    width = int(w*factor)  
    height = int(h*factor)  
    return pil_image.resize((width, height), Image.ANTIALIAS) 
def ch(x):
    if(x<=9):return chr(48+x)
    return chr(65+x-10)
def calc_color(x):
    v1=(255-150)//6*(6-x)
    v2=255//6*(6-x)
    a=(255-v1)//16
    b=(255-v1)%16
    c=(255-v2)//16
    d=(255-v2)%16
    return '#'+ch(a)+ch(b)+ch(c)+ch(d)+'00'



class Viewer(tk.Tk):

    def __init__(self, evn):
        print('jb')
        self.evn=evn
        print('jb')
        super(Viewer, self).__init__()
        print('jb')
        self.origin = np.array([20, 20])
        print('jb')

        self.walls_image=ImageTk.PhotoImage(Image.open('walls.png'))
        self.boxes_image=ImageTk.PhotoImage(Image.open('boxes.png'))
        self.item_fig={}
        self.item_fig['*']=ImageTk.PhotoImage(resize(Image.open('item1.png')))
        self.item_fig['^']=ImageTk.PhotoImage(resize(Image.open('item2.png')))
        self.bomb_pic=ImageTk.PhotoImage(resize(Image.open('bomb.png')))

        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        self.init_image()
        
    def Text(self,xy,v):
        t=self.canvas.create_text(xy[0]*UNIT+20,xy[1]*UNIT+8,anchor='center',text=str(v))
        return t
    def render(self):
        self.reset()
        self.title('Crazy Arcade')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        
        
        self.update()
    
    def init_image(self):
        self.maze=self.evn.maze
        self.players=self.evn.players
        self.players[0].fig=ImageTk.PhotoImage(resize(Image.open('player0.png')))
        self.players[1].fig=ImageTk.PhotoImage(resize(Image.open('player1.png')))
        self.inpulse=self.evn.inpulse
        self.boxes_xyk=self.evn.boxes_xyk
        #self.bombs_data
        #self.bombs_xy

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)
        
        # create origin
        origin = self.origin
        
        #create wall,box,item,bombs
        boxes_xy=[]
        for box in self.boxes_xyk:
            boxes_xy.append((box[0],box[1]))
        for x in range(0,MAZE_W):
            for y in range(0,MAZE_H):
                if (self.maze[x][y]=='#'):
                    wall_now = origin*2 + np.array([UNIT*x+2, UNIT*y+2])
                    self.walls = self.canvas.create_image(wall_now[0],
                        wall_now[1], anchor='se',
                        image=self.walls_image)
                position = self.origin*2 + np.array([UNIT*x+2, UNIT*y+2])
                if (x,y) in boxes_xy:
                    self.canvas.create_image(position[0],
                        position[1], anchor='se',
                        image=self.boxes_image)
                else:
                    if(self.maze[x][y] in ['*','^']):
                        self.canvas.create_image(position[0],
                            position[1], anchor='se',
                            image=self.item_fig[self.maze[x][y]])
                if (x,y) in self.evn.bombs_xy:
                    self.canvas.create_image(origin[0]+x*UNIT,origin[1]+y*UNIT, anchor='center', image=self.bomb_pic)
        
        # create players
        for player_id in range(2):
            xy_now=self.players[player_id].xy
            player_now = origin + np.array([UNIT*xy_now[0], UNIT*xy_now[1]])
            self.players[player_id].players = self.canvas.create_image(player_now[0], \
                player_now[1], anchor='center', \
                image=self.players[player_id].fig)
            
        #create inpulse
        for i in range(MAZE_W):
            for j in range(MAZE_H):
                if(self.inpulse[i][j]==1):
                    position = self.origin + np.array([UNIT*i, UNIT*j])
                    
                    self.inpulse.append(self.canvas.create_rectangle(
                    position[0] - 10, position[1] - 10,
                    position[0] + 10, position[1] + 10,
                    fill="#FF4444"))

        # create text for bombs
        for i in range(self.evn.bombs_cnt):
            self.Text(self.evn.bombs_xy[i],self.evn.bombs_data[i][0])
        
        # pack all
        self.canvas.pack()
        
    def reset(self):
        self.canvas.delete("all")
        self.init_image()
    
        
class Maze(object):
    viewer=None
    def __init__(self, Map):
        #initialize
        self.accu_value=[0]*2
        self.remain_round=GAME_ROUND
        self.origin_map=Map
        global MAZE_H
        MAZE_H=len(Map)
        global MAZE_W
        MAZE_W=len(Map[0])
        self.width=MAZE_W
        self.height=MAZE_H
        self.Map=Map
        super(Maze, self).__init__()
        
        self.init_data(Map)
    
    def render(self):
        #time.sleep(0.1)
        if(self.viewer==None):
            self.viewer=Viewer(self)
        self.viewer.render()
        return self.viewer
    
    def reset(self):
        self.__init__(self.origin_map)
        return state_creator.Get_State(self)
        
    def init_data(self,Map):

        self.action_space = ['u', 'd', 'l', 'r','s','b']
        self.n_actions = len(self.action_space)
        
        self.bombs_cnt=0
        self.bombs_xy=[]
        self.bombs_data=[]  # bomb_data = (time,distance,id)

        self.item_icon={}
        
        self.players=[player(),player()]
        
        self.walls_xy=[]
        
        #maze data
        #maze stores walls('#')/boxes('o')/items('*'/'^')/vacant(' ')/boom('b')
        self.maze=[([' '] * MAZE_H) for i in range(MAZE_W)]
        #bomb's range
        self.maze_=[([0] * MAZE_H) for i in range(MAZE_W)]
		
        #not important
        self.temp=[([0] * MAZE_H) for i in range(MAZE_W)]
        self.time_stamp=1
		
        #visiable inpulse
        self.inpulse=[([0] * MAZE_H) for i in range(MAZE_W)]
        
        #box data
        #(x,y,k) -- position (x,y), item k
        self.boxes_xyk=[]
        self.boxes_cnt=0
        for i in range(MAZE_W):
            for j in range(MAZE_H):
                if(Map[j][i]=='#'):
                    self.walls_xy.append((i,j))
                    self.maze[i][j]='#'
                if(Map[j][i] in ['*','^','o']):
                    ch=Map[j][i]
                    if(ch=='o'):ch=' '
                    self.boxes_cnt+=1
                    self.boxes_xyk.append((i,j,ch))
                    self.maze[i][j]='o';
                if(Map[j][i]=='0'):
                    self.players[0].xy=(i,j)
                if(Map[j][i]=='1'):
                    self.players[1].xy=(i,j)
        

    def step(self,actions):
        '''
        action= 0 1 2 3 4 5
                l,r,u,d,b,s
        '''
        self.remain_round-=1
        value=[0,0]
        #agent move
        for A in actions:
            player_id=A[0]
            action=A[1]
            s=self.players[player_id].xy
            base_action = np.array([0, 0])
            
            # move action
            if action == 2:   # up
                if s[1] > 0:
                    base_action[1] -= 1
            if action == 3:   # down
                if s[1] < MAZE_H -1:
                    base_action[1] += 1
            if action == 1:   # right
                if s[0] < MAZE_W - 1:
                    base_action[0] += 1
            if action == 0:   # left
                if s[0] > 0:
                    base_action[0] -= 1
            
            # bomb action
            last_xy=(self.players[player_id].xy[0],self.players[player_id].xy[1])
            if action == 4:
                if(self.players[player_id].boom_num+1<=self.players[player_id].max_boom_num):
                    self.players[player_id].boom_num+=1
                    self.bombs_cnt+=1
                    self.bombs_data.append((BOOM_T,self.players[player_id].boom_r,player_id))
                    self.bombs_xy.append((last_xy[0],last_xy[1]))
                    self.maze[last_xy[0]][last_xy[1]]='b'
                    self.maze_[last_xy[0]][last_xy[1]]=max(self.maze_[last_xy[0]][last_xy[1]],self.players[player_id].boom_r)
                    value[player_id] += REWARD_BOMB
            else:
                # update position
                new_xy=(last_xy[0]+base_action[0],last_xy[1]+base_action[1])
                if (self.maze[new_xy[0]][new_xy[1]] in ['*','^',' ']):
                    if(self.maze[new_xy[0]][new_xy[1]]=='^'):
                        self.players[player_id].max_boom_num+=1
                        
                        value[player_id] += REWARD_ITEM
                    else:
                        if(self.maze[new_xy[0]][new_xy[1]]=='*'):
                            self.players[player_id].boom_r+=1
                            value[player_id] += REWARD_ITEM
                    self.players[player_id].xy=new_xy

        #clear the item
        def Clear_item(u):
            x,y=u
            if(self.maze[x][y] in ['*','^']):
                self.maze[x][y]=' '
        #player get the item, so clear
        Clear_item(self.players[0].xy)
        Clear_item(self.players[1].xy)
        
        #boom explosion
        bfs=[]
        for i in range(self.bombs_cnt):
            if(self.bombs_data[i][0]==0):
                bfs.append((self.bombs_xy[i],self.bombs_data[i][2]))
        self.time_stamp+=1
        tt=self.time_stamp
        time_att={}
        if(len(bfs)!=0):
            l=0
            r=len(bfs)-1
            dd=[(0,1),(0,-1),(-1,0),(1,0)]
            while(l<=r):
                ((x,y),player_id)=bfs[l]
                self.temp[x][y]=tt
                time_att[(x,y,player_id)]=1
                l+=1
                for d in dd:
                    xx,yy=x,y
                    for j in range(self.maze_[x][y]):
                        xx+=d[0]
                        yy+=d[1]
                        if(xx<0 or yy<0 or xx>=MAZE_W or yy>=MAZE_H or self.maze[xx][yy]=='#'):
                            break
                        self.temp[xx][yy]=tt
                        time_att[(xx,yy,player_id)]=1
                        if(self.maze[xx][yy]=='b'):
                            for i in range(self.bombs_cnt):
                                if(self.bombs_xy[i]==(xx,yy)):
                                    r+=1
                                    bfs.append(((xx,yy),self.bombs_data[i][2]))
                            self.maze[xx][yy]=' '
                        if(self.maze[xx][yy]=='o'):
                            break
        i=0
        while(i<self.bombs_cnt):
            x,y=self.bombs_xy[i]
            if(self.temp[x][y]==tt):
                self.players[self.bombs_data[i][2]].boom_num-=1
                self.maze[x][y]=' '
                self.maze_[x][y]=0
                del self.bombs_xy[i]
                del self.bombs_data[i]
                self.bombs_cnt-=1
            else:
                k1,k2,k3=self.bombs_data[i]
                self.bombs_data[i]=(k1-1,k2,k3)
                i+=1
        i=0
        while(i<self.boxes_cnt):
            x,y,k=self.boxes_xyk[i]
            if(self.temp[x][y]==tt):
                self.boxes_cnt-=1
                ch=self.boxes_xyk[i][2]
                self.maze[x][y]=ch
                if((x,y,0) in time_att):
                    value[0] += REWARD_BOX
                if((x,y,1) in time_att):
                    value[1] += REWARD_BOX
                del self.boxes_xyk[i]
            else:
                i+=1
        #
        #冲击波,仅用来可视化
        for i in range(MAZE_W):
            for j in range(MAZE_H):
                if(self.temp[i][j]==tt):
                    self.inpulse[i][j]=1
                else: self.inpulse[i][j]=0
        done=0
        #判断死亡
        for i in range(2):
            x,y=self.players[i].xy
            if(self.temp[x][y]==tt):
                value[1-i] += REWARD_KILL
                value[i] -= REWARD_KILL
                done=1
        
        item_cnt=0
        for x in range(0,self.width):
            for y in range(0,self.height):
                if (self.maze[x][y]=='*')or(self.maze[x][y]=='^'):
                    item_cnt+=1
        #done=((len(self.boxes_xyk)==0) and (item_cnt==0))

        STATE = state_creator.Get_State(self)
        self.accu_value[0] += value[0]
        self.accu_value[1] += value[1]

        #PUNISH_PER_ROUND
        value[0]-=PUNISH_PER_ROUND
        value[1]-=PUNISH_PER_ROUND

        #game end
        if(self.remain_round==0):
            '''
            if(self.accu_value[0]>self.accu_value[1]):
                value[0]+=REWARD_WIN
                value[1]-=REWARD_WIN
            if(self.accu_value[1]>self.accu_value[0]):
                value[0]-=REWARD_WIN
                value[1]+=REWARD_WIN
            '''
            value[0] -= PUNISH
            value[1] -= PUNISH
            done=1
        return (STATE,value,done)
        
    def get_Map_size(self): # (width,height)
        return (self.width,self.height)
    
    def get_Walls(self): # [(x1,y1),(x2,y2),...]
        return self.walls_xy
    
    def get_Boxes(self): # [(x1,y1,type1),(x2,y2,type2),...] type= '*' or '^' or ' '
        return self.boxes_xyk
    
    def get_Items(self): # [(x1,y1,type1),(x2,y2,type2),...] type= '*' or '^'
        item=[]
        for x in range(0,self.width):
            for y in range(0,self.height):
                if (self.maze[x][y]=='*')or(self.maze[x][y]=='^'):
                    item.append((x,y,self.maze[x][y]));
        return item
        
    def get_Bombs(self): # [bomb1,bomb2,...]  bomb=(x,y,time,distance,id)
        bomb=[]
        for i in range(0,self.bombs_cnt):
            xy=self.bombs_xy[i]
            data=self.bombs_data[i]
            bomb.append((xy[0],xy[1],data[0],data[1],data[2]))
        return bomb
    
    def get_Player(self,player_id): # (pos,max_bomb_num,bomb_num,bomb_r) pos=(x,y)
        return (self.players[player_id].xy,self.players[player_id].max_boom_num,self.players[player_id].boom_num,self.players[player_id].boom_r)
        
    def get_Maze(self): # maze stores walls('#')/boxes('o')/items('*'/'^')/vacant(' ')/boom('b') no player
        return self.maze
        
