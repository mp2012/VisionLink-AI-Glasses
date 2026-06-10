import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')

print("======== 系统可用语音库列表 ========")
for index, voice in enumerate(voices):
    print(f"索引: {index} | 名称: {voice.name} | 语言: {voice.languages}")
print("====================================")

engine.say("测试语音播报，如果能听到这句话，说明系统引擎正常")
engine.runAndWait()