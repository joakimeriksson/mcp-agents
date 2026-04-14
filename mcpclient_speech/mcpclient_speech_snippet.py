import re

def extract_dialog_messages(messages):
    return [ msg for msg in messages if (msg['role'] == 'user' if type(msg)==dict else msg.message.content) ]

def distill_user_info(messages):
    prompt = {'role':'user', 'content':'Summarize the above conversation in this form about the user, leaving fields blank if no information:\nName: \nLanguage: \nPreferences: \n'}
    response = openai.chat.completions.create(
        model=model,
        messages=messages + [prompt],
    )
    return response.choices[0].message.content

def extract_value(key, info):
    reg = "[-+ *#]*" + key + "[-+ *#]*"
    lst = info.split("\n")
    for s in lst:
        m = re.match(reg, s)
        if m:
            return s[m.end():]
    return None

def extract_language(info):
    languages = {"English": "en",
                 "Swedish": "sv",
                 "Svenska": "sv",
                 "German": "de",
                 "Deutch": "de",
                 "French": "fr",
                 "Française": "fr",
                 "Francaise": "fr",
                 "Spanish": "es",
                 "Espanol": "es",
                 "Español": "es"}
    s = extract_value("Language:", info)
    if s:
        for l in languages:
            if re.search(l, s):
                return languages[l]
    return "en"


