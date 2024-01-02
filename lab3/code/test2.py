my_set = set()

def insert_tuple(i, j):
    my_set.add((i, j))
    my_set.add((j, i))

insert_tuple(1, 2)
insert_tuple(3, 4)
insert_tuple(2, 1)


has_calcium = (i, j) in my_set
