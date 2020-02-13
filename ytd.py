from pytube import YouTube
import _thread
import time

def YTdownload(url,i):
        
        yt = YouTube('http://youtube.com/watch?v='+url)
        yt.streams.filter(subtype='mp4').first().download('mp4/') #.filter(only_audio=True)
        print(i,': Finished!')
        
ids=open("rerated_video_ids.txt", "r").readlines()

num=len(ids)
for i in range(num):
    ids[i] = ids[i].strip('\n')  #去掉列表中每一个元素的换行符

for i in range(20):
    #print(i)
    _thread.start_new_thread(YTdownload,(ids[i],i,))
    time.sleep(5)

time.sleep(100)

     

