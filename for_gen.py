
#generator
def generator(s, e):
    i = s
    while i < e:
        yield i 
        i += 1


#generator1
def generator1(s, e):
    i = s
    while i > e:        
        yield i 
        i -= 1

