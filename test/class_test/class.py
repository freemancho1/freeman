class SimpleA:
    
    def __init__(self, *a1, **a2):
        # print(type(a1))
        # print(a1[0], a1[1])
        # print(type(a2))
        # print(a2["aa"])
        # print(a2["aaa"])
        for a in a1:
            print(a)
        
    def aaaaa(self, aaa="a", bbb="b"):
        print(aaa, bbb)
        
        
if __name__ == "__main__":
    my_a = {
        "aaa": "aaaaa", 
        "bbb": "bbbb",
    }

    a = SimpleA(3, 2, aa="aaaa", bb="bbbbb", **my_a)
    a.aaaaa(**my_a)