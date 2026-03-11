import pandas as pd
from pathlib import Path
import json, os, argparse, glob

def json_to_parquet(json_folder, parquet_file):
    alldata = {}
    json_files = glob.glob(f"{json_folder}/*.json")
    for f in json_files:
        try:
            stm = Path(f).stem
            with open(f, 'r') as f_:
                data = json.load(f_)        
                alldata[stm] = data                
        except Exception as e:
            print(f"Error processing {f}: {e}")
            return
        
    
    for f in json_files:os.remove(f)        
    df = pd.DataFrame.from_dict(alldata, orient='index')
    #print(df.head(5))
    df.to_parquet(f"{json_folder}/{parquet_file}")



def txt_to_parquet(txt_folder, parquet_file, ext):
    df_list = []
    txt_files = glob.glob(f"{txt_folder}/*{ext}")
    print(len(txt_files))
    for f in txt_files:
        try:
            temp_df = pd.read_csv(f, sep=r'\s+', header=None)            
            
            # This automatically finds how many data columns you have
            data_cols = [c for c in temp_df.columns]
            temp_df[data_cols] = temp_df[data_cols].apply(pd.to_numeric, errors='coerce')
            temp_df['filename'] = Path(f).stem
            
            df_list.append(temp_df)
        except Exception as e:
            print(f"Error processing {f}: {e}")
            return
        
    df = pd.concat(df_list, ignore_index=True)
    df.to_parquet(f"{txt_folder}/{parquet_file}", index=False, engine='pyarrow')
            
    for f in txt_files: os.remove(f)
    print(f"Successfully created {parquet_file} from log files.")

def load_parquet(parquet_file):
    df = pd.read_parquet(parquet_file)
    print(df.head(5))
    print(df.iloc[0])
    return df

parser = argparse.ArgumentParser()
parser.add_argument("--json", default="")
parser.add_argument("--parquet", default="logs.parquet")
parser.add_argument("--txt", default="")
parser.add_argument("--ext", default=".log")
args = parser.parse_args()


if args.json!="":
    json_to_parquet(args.json, args.parquet)
    load_parquet(args.json + "/" + args.parquet)

elif args.txt!="":
    txt_to_parquet(args.txt, args.parquet, args.ext)
    load_parquet(args.txt + "/" + args.parquet)

#USAGE instructions, important files get deleted
#RAW JSON/TXT DATA  2 parquet
#python DATASET\NETWORK\parquet.py --json "DATASET\\NETWORK\\3Glogs" --parquet logs.parquet
#python DATASET\NETWORK\parquet.py --txt "DATASET\\NETWORK\\norway" --parquet logs.log.parquet

#FORMATTED MAHIMAHI 2 parquet
#python DATASET\NETWORK\parquet.py --ext "" --txt "DATASET\\NETWORK\\fcc" --parquet logs.mahi.parquet
#python DATASET\NETWORK\parquet.py --ext "" --txt "DATASET\TRACES\norway_mahimahi" --parquet logs.mahi.parquet

#FORMATTED RESULTS 2 parquet
#python DATASET\NETWORK\parquet.py --ext "" --txt "DATASET\\TRACES\\fcc_netllm_only" --parquet results.llm.parquet
#python DATASET\NETWORK\parquet.py --ext ".txt" --txt "DATASET\\artifacts\\norway" --parquet results.all.parquet
