from pathlib import Path

base_path = Path('/Users/antonfirc/WebstormProjects/ms_speech_api/Graphs')
gen_file_en = open(base_path.joinpath('ENG-100/result-G-complete.txt'), 'r')
imp_file_en_1 = open(base_path.joinpath('ENG-100/result-D-5k-complete.txt'), 'r')
imp_file_en_2 = open(base_path.joinpath('ENG-100/result-D-1k-complete.txt'), 'r')

gen_file_cs = open(base_path.joinpath('CS/result-G-cs.txt'), 'r')
imp_file_cs_1 = open(base_path.joinpath('CS/result-D-cs5k.txt'), 'r')
imp_file_cs_2 = open(base_path.joinpath('CS/result-D-cs1k.txt'), 'r')

th = 0.3

gen_accept_en = 0
gen_total_en = 0
imp_reject_en = 0
imp_total_en = 0

for line in gen_file_en:
    if float(line) >= 0.3:
        gen_accept_en += 1
    gen_total_en += 1

for line in imp_file_en_1:
    if float(line) < 0.3:
        imp_reject_en += 1
    imp_total_en += 1

for line in imp_file_en_2:
    if float(line) < 0.3:
        imp_reject_en += 1
    imp_total_en += 1

gen_perc_en = (gen_accept_en / gen_total_en)*100
imp_perc_en = (imp_reject_en / imp_total_en)*100

print(f"Genuine accept EN: {gen_accept_en} / {gen_total_en} ... {gen_perc_en}")
print(f"Impostor reject EN: {imp_reject_en} / {imp_total_en} .. {imp_perc_en}")

gen_accept_cs = 0
gen_total_cs = 0
imp_reject_cs = 0
imp_total_cs = 0

for line in gen_file_cs:
    if float(line) >= 0.3:
        gen_accept_cs += 1
    gen_total_cs += 1

for line in imp_file_cs_1:
    if float(line) < 0.3:
        imp_reject_cs += 1
    imp_total_cs += 1

for line in imp_file_cs_2:
    if float(line) < 0.3:
        imp_reject_cs += 1
    imp_total_cs += 1

gen_perc_cs = (gen_accept_cs / gen_total_cs)*100
imp_perc_cs = (imp_reject_cs / imp_total_cs)*100

print(f"Genuine accept CS: {gen_accept_cs} / {gen_total_cs} ... {gen_perc_cs}")
print(f"Impostor reject CS: {imp_reject_cs} / {imp_total_cs} .. {imp_perc_cs}")

gen_accept_total = gen_accept_en + gen_accept_cs
gen_total_total = gen_total_en + gen_total_cs
imp_reject_total = imp_reject_cs + imp_reject_en
imp_total_total = imp_total_en + imp_total_cs

gen_perc_total = (gen_accept_total / gen_total_total)*100
imp_perc_total = (imp_reject_total / imp_total_total)*100

print(f"Genuine accept total: {gen_accept_total} / {gen_total_total} ... {gen_perc_total}")
print(f"Impostor reject total: {imp_reject_total} / {imp_total_total} .. {imp_perc_total}")

# Eng
# Genuine accept: 1052 / 1195 ... 88.03347280334728
# Impostor reject: 2575 / 2697 .. 95.47645532072674

# cs
# Genuine accept: 537 / 626 ... 85.78274760383387
# Impostor reject: 2674 / 3048 .. 87.72965879265092

#  total
# Genuine accept: 1589 / 1821 ... 87.26
# Impostor reject: 5249 / 5745 .. 91.37