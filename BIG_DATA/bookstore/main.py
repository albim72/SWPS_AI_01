from book import Book
# import book

bk = Book(3,"Wiedźmin","Andrzej Sapkowski",42)
print(bk)
print(bk.oprawa)
bk(21)

bk.setid(88)
print(bk.getid())

print(bk.cena)

bk.cena = 61
print(bk.cena)
