import os
from ffmpy3 import FFmpeg

#filepath = r"/Users/brucepk/test"   
#os.chdir(filepath)                  
filenames = os.listdir('mp4')
num = len(filenames)
ii=936
for i in range(937,num):
    print('now is:',i)
    changefile = 'mp4/'+filenames[i]
    #print(changefile[-3:])
    if changefile[-4:] != '.mp4':
        continue
    outputfile = '44100/'+str(ii)+'.wav'
    ii+=1
    ff = FFmpeg(
                inputs={changefile: None},
                outputs={outputfile: '-y -vn -ar 44100 -ac 1 -ab 192 -f wav'}  #1:2
                )
    #print(ff.cmd)
    ff.run()