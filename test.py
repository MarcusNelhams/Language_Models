import re

with open('1b_benchmark.train.tokens', 'r', encoding='utf8') as file:
    lines = file.readlines()

start_count = 0
stop_count = 0
p_count = 0
h_count = 0
tot = 0
for line in lines:
    start_count += 1
    stop_count += 1
    l = re.split('\s+', line)
    if '.' in l:
        p_count += 1
    if 'HDTV' in l:
        h_count += 1
    tot += len(line)

print(start_count)
print(stop_count)
print(p_count)
print(h_count)
print(tot)