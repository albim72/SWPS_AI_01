#komentarz jednoliniowy
"""
komentarz wieloliniowy - dokumentacyjny
druga
trzecia
"""
import math


#przykład 1
def fx(n):
    return n**7


n=100
def policz(a:int,b:int,c:float=4,y:float=2)->float:
    global n
    n = (a+b)*y-c+n+fx(a)
    return n

print(policz(6,3,7,2))
print(policz(6,3,4.4,0.11))
print(policz(6,True,4.4,False))
print(policz(1,1,8))
print(policz(1,1))
print(policz(1,1,11))
print(policz(8.8,1,11))
# print(policz(8.8,"4567",33))
print(n)

#orzykład 2 - funkcje anonimowe

# def zwykla(e,g):
#     return 2*e+g

print((lambda e,g:2*e+g)(3.4,7))
b = lambda v:v*6

print(b(45))

def multi(n):
    return lambda a:a*n

print(multi(7)(2))

liczby = [67,2,-6,0,24,199,900,-425,24,7,-234,-4,378]

nparz = list(filter(lambda x:x%2==0,liczby))
print(nparz)

cube = list(map(lambda x:x**3,liczby))
print(cube)

#list cmprehension

speclista = [math.sqrt(x)*(x+7) for x in range(1,10_000_000) if x%2==0]
print(sum(speclista))
