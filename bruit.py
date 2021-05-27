import random


f = open("raw_data.txt", 'r')
linewrite = []
lines = f.readlines()
for line in lines:
    split = line.split(' ')
    if split[-1].find("1"):
        split_copy = ''
        for num in split[:-1]:
            if random.randint(0, 1) == 0:
                split_copy += num + ' '
            else:
                num = float(num)
                rand = random.uniform(0.1, 0.5)
                if random.randint(0, 1) == 0:
                    num -= rand
                else:
                    num += rand
                num = round(num, 3)
                num = str(num)
                split_copy += num + ' '
        split_copy += split[-1]
        linewrite.append(split_copy)
w = open("noise.txt", 'w')
w.writelines(linewrite)
