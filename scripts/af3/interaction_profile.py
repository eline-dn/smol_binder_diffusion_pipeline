# parse the xml outputs from plip
import pandas as pd
import xml.etree.ElementTree as ET
import glob
import os
df=pd.read_csv("selected_scores.csv")
for xml_file in glob.glob(f"*_report.xml"):
  id=xml_file.replace("_model_report.xml", "")
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
df.to_csv("metrics_hbonds_counts.csv")



  
