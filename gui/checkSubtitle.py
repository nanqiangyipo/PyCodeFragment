import os

# 翻译源文件
word = r'D:\workingDirectory\pythonProjects\CodeFragmentPy\gui\139422051_139433106.txt'
# 结果保存文件
save = r'D:\workingDirectory\pythonProjects\CodeFragmentPy\gui\mark.txt'
# 音频目录
audio = r'G:\audio'


solved=[]

if os.path.exists(save):
    solved_file = open(save, mode='r', encoding='utf8')
    old = solved_file.readlines()
    if len(old)>0:
        old = [ s.split('@@')[0] for s in old ]
        solved.extend(old)
    solved_file.close()

word_open = open(word ,mode='r', encoding='utf8')
save_write = open(save, mode='a', encoding='utf8')

for line in word_open:
    id, cont = line.split('\t')
    if id in solved:
        continue
    cont=eval(cont)
    print('begin {}'.format(id))
    for text in cont:
        bt = divmod(text['begin_time']//1000, 60)
        et = divmod(text['end_time']//1000, 60)
        print('%d:%d-%d:%d' % (bt[0],bt[1],et[0],et[1]), text['text'])
    if os.path.exists(os.path.join(audio,id+'.aac')):
        os.startfile(os.path.join(audio,id+'.aac'))
    ctl = input('\n回车标1：')
    if ctl=='' or ctl=='1':
        save_write.write('{}@@1\n'.format(id))
    else:
        save_write.write(('{}@@{}\n'.format(id,ctl)))
    save_write.flush()
    print('end \n\n')

save_write.close()
word_open.close()

# 1可用
# 2可接受
# 3粤语
# 4有背景音的说话，可以用来验证音频与视频结合的效果对比