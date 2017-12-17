class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."
    def __init__(self):
        self.list = []

    def push(self,item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0,item)

    def pop(self):
        """
          Dequeue the earliest enqueued item still in the queue. This
          operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0

class baseline():
    def BFS(queue,dist,w,h):
        while True:
            if queue.isEmpty():
                break
            x,y=queue.pop()
            if x>0 and (not((x-1,y) in dist)):
                dist[(x-1,y)]=dist[(x,y)]+1
                queue.push((x-1,y))
            if x<w-1 and (not((x+1,y) in dist)):
                dist[(x+1,y)]=dist[(x,y)]+1
                queue.push((x+1,y))
            if y>0 and (not((x,y-1) in dist)):
                dist[(x,y-1)]=dist[(x,y)]+1
                queue.push((x,y-1))
            if y<h-1 and (not((x,y+1) in dist)):
                dist[(x,y+1)]=dist[(x,y)]+1
                queue.push((x,y+1))
    
    def dist_direction(pos,dist):
        dis=dist[pos]
        action='s'
        
        x,y=pos[0]-1,pos[1]
        if((x,y) in dist) and dist[(x,y)]<dis:
            dis=dist[(x,y)]
            action='l'
        x,y=pos[0]+1,pos[1]
        if((x,y) in dist) and dist[(x,y)]<dis:
            dis=dist[(x,y)]
            action='r'
        x,y=pos[0],pos[1]-1
        if((x,y) in dist) and dist[(x,y)]<dis:
            dis=dist[(x,y)]
            action='u'
        x,y=pos[0],pos[1]+1
        if((x,y) in dist) and dist[(x,y)]<dis:
            dis=dist[(x,y)]
            action='d'
        
        return action
    
    def choose_action(env,now_id):
        return 's'
        w,h=env.get_Map_size()
        #pos=env.players_xy[now_id]
        pos=env.get_Player(now_id)[0]
        walls=env.get_Walls()
        bombs=env.get_Bombs()
        ban=[]
        boxes=[]
        item=[]
        maze=env.get_Maze()
        
        for A in env.get_Items():
            item.append((A[0],A[1]))
        
        # for x in range(0,w):
            # for y in range(0,h):
                # if (env.maze[x][y]=='*')or(env.maze[x][y]=='^'):
                    # item.append((x,y));
        
        for A in env.get_Boxes():
            boxes.append((A[0],A[1]))
        
        for i in range(0,w):
            walls.append((i,-1))
            walls.append((i,h))
        for i in range(0,h):
            walls.append((-1,i))
            walls.append((w,i))
        
        # for i in range(0,env.bombs_cnt):
            # bombs.append((env.bombs_xy[i][0],env.bombs_xy[i][1],env.bombs_data[i][0],env.bombs_data[i][1]))
            
        for b in bombs:
            if not ((b[0],b[1]) in ban):
                ban.append((b[0],b[1]))
            x,y=b[0],b[1]
            for j in range(0,b[3]):
                x-=1
                if ((x,y) in walls)or((x,y) in boxes):
                    break
                if not ((x,y) in ban):
                    ban.append((x,y))
            x,y=b[0],b[1]
            for j in range(0,b[3]):
                x+=1
                if ((x,y) in walls)or((x,y) in boxes):
                    break
                if not ((x,y) in ban):
                    ban.append((x,y))
            x,y=b[0],b[1]
            for j in range(0,b[3]):
                y-=1
                if ((x,y) in walls)or((x,y) in boxes):
                    break
                if not ((x,y) in ban):
                    ban.append((x,y))
            x,y=b[0],b[1]
            for j in range(0,b[3]):
                y+=1
                if ((x,y) in walls)or((x,y) in boxes):
                    break
                if not ((x,y) in ban):
                    ban.append((x,y))
        
        dist={}
        dist2={}
        for A in walls:
            dist[A]=100000000
            dist2[A]=100000000
        for A in ban:
            dist[A]=10000000
        for A in bombs:
            dist[(A[0],A[1])]=100000000
            dist2[(A[0],A[1])]=100000000
        for A in boxes:
            dist[A]=100000000
            dist2[A]=100000000
        
        queue=Queue()
        for x in range(0,w):
            for y in range(0,h):
                if (not((x,y) in walls)) and (not((x,y) in ban)) and ((not((x,y) in boxes))):
                    dist[(x,y)]=0
                    dist2[(x,y)]=0
                    queue.push((x,y))
        baseline.BFS(queue,dist2,w,h)
        
        queue=Queue()
        for x in range(0,w):
            for y in range(0,h):
                if (not((x,y) in walls)) and (not((x,y) in ban)) and ((not((x,y) in boxes))):
                    dist[(x,y)]=0
                    queue.push((x,y))
        baseline.BFS(queue,dist2,w,h)
        
        for A in ban:
            if A in dist2:
                dist[A]+=dist2[A]
            else:
                dist[A]+=10000000000
        
        if dist[pos]!=0:
            action=baseline.dist_direction(pos,dist)
            
            x,y=pos
            
        else:
            dist3={}
            for A in walls:
                dist3[A]=100000000
            for A in ban:
                dist3[A]=10000000
                
            queue=Queue()
            for x in range(0,w):
                for y in range(0,h):
                    if (not((x,y) in walls)) and (not((x,y) in ban)) and ((maze[x][y]=='o')or((x,y)in item)):
                        dist3[(x,y)]=0
                        queue.push((x,y))
            baseline.BFS(queue,dist3,w,h)
            
            if not (pos in dist3):
                action='s'
            else:
                if dist3[pos]==1:
                    action='b'
                    x,y=pos
                    if (x-1,y) in item:
                        action='l'
                    if (x+1,y) in item:
                        action='r'
                    if (x,y-1) in item:
                        action='u'
                    if (x,y+1) in item:
                        action='d'
                else:
                    action=baseline.dist_direction(pos,dist3)
            
        return action
