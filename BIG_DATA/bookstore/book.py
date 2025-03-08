class Book:
    #definicja stanu

    # def __new__(cls, *args, **kwargs):
    #     return object.__new__(Book)

    def __init__(self,id,tytul,autor,cena=20):
        self.idbook = id
        self.tytul = tytul
        self.autor = autor
        self.cena = cena
        self.oprawa = "miękka"
        self.create_book()

    #definicja zachowania
    def create_book(self):
        print(f"utworzono ksiązkę - nowy obiekt klasy {self.__class__.__qualname__}")

