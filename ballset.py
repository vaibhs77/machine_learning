from sklearn import tree
def marvellousml(weight,surface):
    Ballfeature=[[35,1],[47,1],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]]
    names = [1,1,1,2,1,2,1,1,1,2,1,2,1,2]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(Ballfeature,names)
    result = clf.predict([[weight,surface]])
    if result ==1:
        print("your object is tennis ball")
    elif result == 2:
        print("your object look like cricket ball"
            )
def main():
    print("_____________________ball prdection using ml _________________")
    print("enter the weight of object")
    weight = input()
    print("what is the surface type rough or smooth")
    surface = input()
    if surface.lower()=="rough":
     surface = 1
    elif surface.lower()=="smooth":
        surface= 0



        exit()
    marvellousml(weight,surface)
if __name__=="__main__":
    main()
