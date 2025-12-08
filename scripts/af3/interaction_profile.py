# parse the xml outputs from plip
import pandas as pd
import xml.etree.ElementTree as ET
import glob
import os
df=pd.read_csv("selected_scores.csv")
for xml_file in glob.glob(f"*_report.xml"):
  id=xml_file.replace("_report.xml", "")
  tree = ET.parse(xml_file)
  root = tree.getroot()
  countB=0
  for hb in root.iter('hydrogen_bond'):
    print(hb.attrib, hb.tag)
    ch=hb.find('reschain').text
    #print(ch)
    if ch=="B":
      countB+=1
  #df=pd.df({"design":design, "hb_count"=countB})
  id_index=df['id']==id
  df.loc[id_index, "hb_count"] = countB
  #df.at[id_index, "hb_count"]=countB
  #header=os.path.exists("hbonds_counts.csv")
df_clean=df.drop_duplicates(subset="binder_sequence", keep="first")
df_clean.to_csv("selected_scores.csv", index=False)

subset = df_clean[df_clean["hb_count"] >= 1].sort_values(by="sc", ascending=False)
subset.to_csv("top_hb_scores.csv")

import shutil

df_clean=pd.read_csv("selected_scores.csv")
os.makedirs("sc_cutoff", exist_ok=True)
subset = df_clean[df_clean["sc"] >= 0.6].sort_values(by="sc", ascending=False)
for id in subset["id"]:
  cif=id + ".cif"
  shutil.copy(cif, os.path.join("sc_cutoff",cif))
subset.to_csv("sc_cutoff/scores.csv")


os.makedirs("hb_cutoff", exist_ok=True)
subset = df_clean[(df_clean["hb_count"] >= 1) & (df_clean["sc"] >= 0.6)].sort_values(by="sc", ascending=False)
for id in subset["id"]:
  cif=id + ".cif"
  shutil.copy(cif, os.path.join("hb_cutoff",cif))
subset.to_csv("hb_cutoff/scores.csv")

  
