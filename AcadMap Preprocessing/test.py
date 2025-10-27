# # for i in range(1, 12):
# #     fx = open(f"/home/amaydixit11/Desktop/{i}.txt")
# #     x = fx.read()
# #     x = x.replace("\n\n", "\t")
# #     x = x.replace("\n", " ")
# #     x = x.replace("\t", "\n")
# #     ffx = open(f"/home/amaydixit11/Desktop/{i*100 + i*10 + i}.txt", "w")
# #     ffx.write(x)

# for i in range(1, 11):
#     fx = open(f"/home/amaydixit11/Desktop/{i*111}.txt")
#     x = fx.readlines()
#     ff = x[0]
#     x = [i.replace(",", ";;") for i in x[1:]]
#     x = [ff.strip() + "," + i[:6] + "," + i[7:] for i in x]
#     # x[0] = ff
#     x = ''.join(x)
#     # f = x.find("\n")
#     # x = x.replace(",", ";;")
#     # x = x[:f+7] + "," + x[f+8:]
#     x = x.replace(" Credits (", ",")
#     x = x.replace(") Prerequisite(s): ", ",")
#     x = x.replace(" Overlap with: ", ",")
#     x = x.replace("NA ", "NA,")
#     x = x.replace("%) ", "%),")
#     x = x.split("\n")
#     for j in x:
#         fff = j.index('-')
#         print(fff)
#         print(j[:fff+1])
#         new = fff-j[:fff][::-1].index(" ")-1
#         tor = x.index(j)
#         x[tor] = j[:new]+","+j[new+1:]
#     x = '\n'.join(x)
#     x = x.replace(";;", ";")
#     x = "Department,course_code,course_name,credits,distribution,prerequisites,overlap_with,syllabus\n" + x
#     ffx = open(f"/home/amaydixit11/Desktop/{i*11111}.txt", "w")
#     ffx.write(x)
#     fx.close()
#     ffx.close()

codes = ["EVL501",
"EVL500",
"EVL502",
"EVL503",
"EVL5xx",
"EVL600",
"CYL100",
"CYL101",
"CYP102",
"CYL400",
"CYL401",
"CYL500",
"CYL501",
"CYP502",
"CYP503",
"CYL504",
"CYL505",
"CYL506",
"CYL507",
"CYL508",
"CYL509",
"CYL510",
"CYL600",
"CYL601",
"CYL602",
"CYL603",
"CYL605",
"CYL610",
"ECL101",
"ECL501",
"ECL502",
"ECL503",
"ECL504",
"ECL511",
"ECL599",
"ECL604",
"PHL101",
"PHP102",
"PHL403",
"PHL404",
"PHL501",
"PHL502",
"PHL505",
"PHP506",
"PHL507",
"PHL508",
"PHL509",
"PHL510",
"PHP511",
"PHL601",
"PHL602",
"PHL604",
"PHL607",
"PHL609",
"PHL610",
"PHL611",
"LAL100",
"LAL101",
"LAN102",
"LAN103",
"LAL201",
"LAL221",
"LAL226",
"LAL247",
"LAN249",
"LAL252",
"LAL731",
"LAL733",
"LAL734",
"LAL735",
"MTL201",
"MTL202",
"MTL301",
"MTP301",
"MTP302",
"MTQ401",
"MTL501",
"MTL602",
"MTL603",
"MTL655",
"MEP102",
"MEL211",
"MEL212",
"MEL214",
"MEL231",
"MEL232",
"MEL251",
"MEL252",
"MEP302",
"MEL304",
"MEL313",
"MEL333",
"MEL334",
"MEL351",
"MEP371",
"MEP376",
"MEP381",
"MEL414",
"MEL501",
"MEL558",
"MEL611",
"MEL612",
"MEL613",
"MEL614",
"MEL622",
"MEL623",
"MEL624",
"MEL631",
"MEL636",
"MEL651",
"MEL633",
"MEL637",
"MEL652",
"MEL655",
"MEL656",
"MEL658",
"MEL659",
"MAL100",
"MAL101",
"MAL400",
"MAL401",
"MAL402",
"MAL403",
"MAL404",
"MAL405",
"MAL406",
"MAL500",
"MAL501",
"MAL502",
"MAL503",
"MAL504",
"MAL505",
"MAL510",
"MAL511",
"MAL512",
"MAL514",
"MAL600",
"MAL602",
"MAL603",
"MAL604",
"MML201",
"MML202",
"MML203",
"MML204",
"MML205",
"MML251",
"MMP251",
"MML252",
"MML253",
"MML254",
"MML301",
"MMP301",
"MML302",
"MMP302",
"MML303",
"MMP303",
"MML351",
"MML401",
"MMP401",
"MML501",
"MMP501",
"MML551",
"MML552",
"MML553",
"MMP553",
"MML554"]

from collections import Counter

duplicates = [course for course, count in Counter(codes).items() if count > 1]
print(duplicates)