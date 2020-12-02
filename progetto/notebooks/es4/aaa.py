b = 1
if b > 0:
    div = 10


    def mean(a):
        index = 0
        le = len(a)
        s = 0
        while index < le:
            s += a[index]
            index += 1
        a[int(le / 2)] = 777
        return float(s) / float(div)  # float(le)


    for i in range(5, 10):
        arr = [x for x in range(1, i + 1)]
        print("\n\n", arr)
        print(mean(arr))
        print(arr)
else:
    print("WHAT")
