# -*- coding: utf-8 -*-


def copy_file():
    i = 4
    while i < 50:
        print(i)
        src = file("data/1/1111.jpeg", "r+")
        i += 1
        des = file("data/1/test0"+str(i)+".jpeg", "w+")
        des.writelines(src.read())
    src.close()
    des.close()


