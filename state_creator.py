import env
import cv2
import numpy as np

ZiTi=0.3

def Resize(img, flag=0):
    if(flag==1):
        return cv2.resize(img,(Unit//2-2,Unit//2-2), interpolation=cv2.INTER_AREA)
    return cv2.resize(img,(Unit-2,Unit-2), interpolation=cv2.INTER_AREA)
Unit=23
walls_image=Resize(cv2.imread('walls.png'))
boxes_image=Resize(cv2.imread('boxes.png'))
item_fig={}
item_fig['*']=Resize(cv2.imread('item1.png'))
item_fig['^']=Resize(cv2.imread('item2.png'))
bomb_pic=Resize(cv2.imread('bomb.png'),1)
P0=Resize(cv2.imread('player0.png'),1)
P1=Resize(cv2.imread('player1.png'),1)
def Get_State(env):
    def create_line(x0, y0, x1, y1):
        for i in range(x0,x1+1):
            for j in range(y0,y1+1):
                result[i,j]=(0,0,0)
    def wph(i, j, k):
        if(i>=0 and i<env.height*Unit and j>=0 and j<env.width*Unit):
            result[i,j]=k
    def Put_Image(img, pos, offset_x, offset_y):
        x,y = img.shape[0],img.shape[1]
        for i in range(x):
            for j in range(y):
                wph(i+pos[0]+offset_x, j+pos[1]+offset_y, img[i,j])
    def Text_bomb(xy, v, dis):
        cv2.putText(result, str(v), (xy[0]*Unit + Unit//8,xy[1]*Unit+Unit*7//8),cv2.FONT_HERSHEY_SIMPLEX, ZiTi, (0,0,0), 1)
        cv2.putText(result, str(dis), (xy[0]*Unit + Unit//8+Unit//2,xy[1]*Unit+Unit*7//8),cv2.FONT_HERSHEY_SIMPLEX, ZiTi, (0,0,0), 1)
    def Text_player(xy, v, flag):
        if(flag==0):
            cv2.putText(result, str(v), (xy[0]*Unit + Unit//8,xy[1]*Unit+Unit*3//8),cv2.FONT_HERSHEY_SIMPLEX, ZiTi, (0,0,0), 1)
        else :
            cv2.putText(result, str(v), (xy[0]*Unit+Unit//2 + Unit//8,xy[1]*Unit+Unit*3//8),cv2.FONT_HERSHEY_SIMPLEX, ZiTi, (0,0,0), 1)
            

    result = np.zeros((env.height*Unit,env.width*Unit,3),np.uint8)
    result[...]=255
    maze=env.maze
    players=env.players
    players[0].fig=P0
    players[1].fig=P1
    inpulse=env.inpulse
    boxes_xyk=env.boxes_xyk
    
    #self.bombs_data
    #self.bombs_xy

    # create grids
    for c in range(0, env.width * Unit, Unit):
        x0, y0, x1, y1 = c, 0, c, env.height * Unit-1
        create_line(y0, x0, y1, x1)
    for r in range(0, env.height * Unit, Unit):
        x0, y0, x1, y1 = 0, r, env.width * Unit-1, r
        create_line(y0, x0, y1, x1)
    
    #create wall,box,item,bombs
    boxes_xy=[]
    for box in boxes_xyk:
        boxes_xy.append((box[0],box[1]))
    for x in range(0,env.width):
        for y in range(0,env.height):
            if (maze[x][y]=='#'):
                wall_now = np.array([Unit*y+1, Unit*x+1])
                Put_Image(walls_image,wall_now, 0, 0)
            position =np.array([Unit*y+1, Unit*x+1])
            if (x,y) in boxes_xy:
                Put_Image(boxes_image, position, 0, 0)
            else:
                if(maze[x][y] in ['*','^']):
                    Put_Image(item_fig[maze[x][y]], position, 0, 0)
            if (x,y) in env.bombs_xy:
                Put_Image(bomb_pic, position, Unit//2, 0)
    
    # create players
    for player_id in range(2):
        xy_now=players[player_id].xy
        player_now = np.array([Unit*xy_now[1]+1, Unit*xy_now[0]+1])
        u,v=0,0
        if(player_id==1):
            u=0
            v=Unit//2
        Put_Image(players[player_id].fig, player_now, u, v)

    
    bombs_t=[([100] * env.height) for i in range(env.width)]
    
    # create text for bombs
    for i in range(env.bombs_cnt):
        xy=env.bombs_xy[i]
        bombs_t[xy[0]][xy[1]]=min(bombs_t[xy[0]][xy[1]], env.bombs_data[i][0])
    
    for x in range(0,env.width):
        for y in range(0,env.height):
            if (maze[x][y]=='b'):
                Text_bomb((x,y) , bombs_t[x][y], env.maze_[x][y])
    
    Text_player(players[0].xy, players[0].max_boom_num-players[0].boom_num, 0)
    Text_player(players[1].xy, players[1].max_boom_num-players[1].boom_num, 1)

    return result
