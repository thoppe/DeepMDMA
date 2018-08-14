import glob, os
import pandas as pd

images_per_line = 6
F_JPG = glob.glob("../results/images/*.jpg")

data = []
for f in sorted(F_JPG):
    name = os.path.basename(f)

    item = {}
    item['f'] = f

    item['channel'] = name.split('_')[0]
    item['n'] = int(name.split('_')[-1].split('.')[0])
    data.append(item)

    
df = pd.DataFrame(data).sort_values("n")

for channel,dfx in df.groupby("channel"):

    f_save = f'display_{channel}.md'
    with open(f_save, 'w') as FOUT:
        FOUT.write(f"# {channel}\n")


        for k in range(df.n.min(), df.n.max(), images_per_line):
            FOUT.write(f"*{channel}:{k} - {channel}{k+images_per_line}*\n")

            for n in range(k, k+images_per_line):
                f_image = f'../results/images/{channel}_3x3_pre_relu_{n}.jpg'
                img = f"![{channel}:{n}]({f_image}) "
                FOUT.write(img)
            FOUT.write('\n\n')
   
    

#print(df)

