import config.config as config

class Direction:
    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


class Snake:
    def __init__(self, body, direction):
        # telo je lista koordinata: [(3,2), (3,1), (3,0)]
        # glava je prvi element liste
        self.body = list(body)
        self.direction = direction
        self.alive = True
        self.grew = False  # ako je zmija jela hranu u ovom potezu, sledeci potez nece ukloniti rep

    def head(self):
        return self.body[0]

    def set_direction(self, new_direction):
        # ne moze se kretati u suprotnom pravcu
        opposite = (-self.direction[0], -self.direction[1])
        if new_direction != opposite:
            self.direction = new_direction

    def move(self):
        '''
        Metoda koja pomera zmiju u trenutnom pravcu. Ako je zmija jela hranu, sledeci potez nece ukloniti rep.
        Ako nije jela hranu, uklanja se poslednji element (rep) zmije.
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
        self.grew = True  # sledeci potez ne uklanja rep

    def get_body(self):
        return self.body
    
def make_snake1():
    return Snake(config.SNAKE1_START, config.SNAKE1_START_DIR)

def make_snake2(): 
    return Snake(config.SNAKE2_START, config.SNAKE2_START_DIR)