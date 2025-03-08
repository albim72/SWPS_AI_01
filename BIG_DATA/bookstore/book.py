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

    def __repr__(self):
        return f"książka [id->{self.idbook}] {self.tytul}, autor: {self.autor}"

    def __call__(self, procent):
        print(f"Rabat przy zakupie: {procent/100 * self.cena} zł")

    #definicja zachowania
    def create_book(self):
        print(f"utworzono ksiązkę - nowy obiekt klasy {self.__class__.__qualname__}")

    def getid(self):
        return self.idbook

    def setid(self,noweid):
        self.idbook = noweid

    @property
    def cena(self):
        return self._cena

    @cena.setter
    def cena(self,ncena):
        self._cena = ncena


