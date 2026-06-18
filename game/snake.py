import config 

class Direction:
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


class Snake:
    def init(self, body, direction):
        # body is coordinates list eg. [(3,2), (3,1), (3,0)]
        # head is list[0]
        self.body = list(body)
        self.direction = direction
        self.alive = True
        self.grew = False  # if it should grow this move

    def head(self):
        return self.body[0]

    def set_direction(self, new_direction):
        # stopping 180 degree turn (can't move directly back)
        opposite = (-self.direction[0], -self.direction[1])
        if new_direction != opposite:
            self.direction = new_direction

    def move(self):
        '''
        Move is implemented by adding a new element to the start of the list 
        That is the new position of the head 
        If the snake grew in this move we don;t remove the last tail element 
        If it didn't grow we remove it
        '''
        head_row, head_col = self.body[0]
        dr, dc = self.direction
        new_head = (head_row + dr, head_col + dc)

        self.body.insert(0, new_head)  

        if self.grew:
            self.grew = False  
        else:
            self.body.pop()  

    def grow(self):
        self.grew = True  # next move() won't remove tail

    def get_body(self):
        return self.body
    
def make_snake1():
    return Snake(config.SNAKE1_START, config.SNAKE1_START_DIR)

def make_snake2(): 
    return Snake(config.SNAKE2_START, config.SNAKE2_START_DIR)