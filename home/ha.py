import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib import style
style.use('ggplot')
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
stop_words = ["0o", "0s", "3a", "3b", "3d", "6b", "6o", "a", "a1", "a2", "a3", "a4", "ab", "able", "about", "above", "abst", "ac", "accordance", "according", "accordingly", "across", "act", "actually", "ad", "added", "adj", "ae", "af", "affected", "affecting", "affects", "after", "afterwards", "ag", "again", "against", "ah", "ain", "ain't", "aj", "al", "all", "allow", "allows", "almost", "alone", "along", "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "announce", "another", "any", "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "ao", "ap", "apart", "apparently", "appear", "appreciate", "appropriate", "approximately", "ar", "are", "aren", "arent", "aren't", "arise", "around", "as", "a's", "aside", "ask", "asking", "associated", "at", "au", "auth", "av", "available", "aw", "away", "awfully", "ax", "ay", "az", "b", "b1", "b2", "b3", "ba", "back", "bc", "bd", "be", "became", "because", "become", "becomes", "becoming", "been", "before", "beforehand", "begin", "beginning", "beginnings", "begins", "behind", "being", "believe", "below", "beside", "besides", "best", "better", "between", "beyond", "bi", "bill", "biol", "bj", "bk", "bl", "bn", "both", "bottom", "bp", "br", "brief", "briefly", "bs", "bt", "bu", "but", "bx", "by", "c", "c1", "c2", "c3", "ca", "call", "came", "can", "cannot", "cant", "can't", "cause", "causes", "cc", "cd", "ce", "certain", "certainly", "cf", "cg", "ch", "changes", "ci", "cit", "cj", "cl", "clearly", "cm", "c'mon", "cn", "co", "com", "come", "comes", "con", "concerning", "consequently", "consider", "considering", "contain", "containing", "contains", "corresponding", "could", "couldn", "couldnt", "couldn't", "course", "cp", "cq", "cr", "cry", "cs", "c's", "ct", "cu", "currently", "cv", "cx", "cy", "cz", "d", "d2", "da", "date", "dc", "dd", "de", "definitely", "describe", "described", "despite", "detail", "df", "di", "did", "didn", "didn't", "different", "dj", "dk", "dl", "do", "does", "doesn", "doesn't", "doing", "don", "done", "don't", "down", "downwards", "dp", "dr", "ds", "dt", "du", "due", "during", "dx", "dy", "e", "e2", "e3", "ea", "each", "ec", "ed", "edu", "ee", "ef", "effect", "eg", "ei", "eight", "eighty", "either", "ej", "el", "eleven", "else", "elsewhere", "em", "empty", "en", "end", "ending", "enough", "entirely", "eo", "ep", "eq", "er", "es", "especially", "est", "et", "et-al", "etc", "eu", "ev", "even", "ever", "every", "everybody", "everyone", "everything", "everywhere", "ex", "exactly", "example", "except", "ey", "f", "f2", "fa", "far", "fc", "few", "ff", "fi", "fifteen", "fifth", "fify", "fill", "find", "fire", "first", "five", "fix", "fj", "fl", "fn", "fo", "followed", "following", "follows", "for", "former", "formerly", "forth", "forty", "found", "four", "fr", "from", "front", "fs", "ft", "fu", "full", "further", "furthermore", "fy", "g", "ga", "gave", "ge", "get", "gets", "getting", "gi", "give", "given", "gives", "giving", "gj", "gl", "go", "goes", "going", "gone", "got", "gotten", "gr", "greetings", "gs", "gy", "h", "h2", "h3", "had", "hadn", "hadn't", "happens", "hardly", "has", "hasn", "hasnt", "hasn't", "have", "haven", "haven't", "having", "he", "hed", "he'd", "he'll", "hello", "help", "hence", "her", "here", "hereafter", "hereby", "herein", "heres", "here's", "hereupon", "hers", "herself", "hes", "he's", "hh", "hi", "hid", "him", "himself", "his", "hither", "hj", "ho", "home", "hopefully", "how", "howbeit", "however", "how's", "hr", "hs", "http", "hu", "hundred", "hy", "i", "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ibid", "ic", "id", "i'd", "ie", "if", "ig", "ignored", "ih", "ii", "ij", "il", "i'll", "im", "i'm", "immediate", "immediately", "importance", "important", "in", "inasmuch", "inc", "indeed", "index", "indicate", "indicated", "indicates", "information", "inner", "insofar", "instead", "interest", "into", "invention", "inward", "io", "ip", "iq", "ir", "is", "isn", "isn't", "it", "itd", "it'd", "it'll", "its", "it's", "itself", "iv", "i've", "ix", "iy", "iz", "j", "jj", "jr", "js", "jt", "ju", "just", "k", "ke", "keep", "keeps", "kept", "kg", "kj", "km", "know", "known", "knows", "ko", "l", "l2", "la", "largely", "last", "lately", "later", "latter", "latterly", "lb", "lc", "le", "least", "les", "less", "lest", "let", "lets", "let's", "lf", "like", "liked", "likely", "line", "little", "lj", "ll", "ll", "ln", "lo", "look", "looking", "looks", "los", "lr", "ls", "lt", "ltd", "m", "m2", "ma", "made", "mainly", "make", "makes", "many", "may", "maybe", "me", "mean", "means", "meantime", "meanwhile", "merely", "mg", "might", "mightn", "mightn't", "mill", "million", "mine", "miss", "ml", "mn", "mo", "more", "moreover", "most", "mostly", "move", "mr", "mrs", "ms", "mt", "mu", "much", "mug", "must", "mustn", "mustn't", "my", "myself", "n", "n2", "na", "name", "namely", "nay", "nc", "nd", "ne", "near", "nearly", "necessarily", "necessary", "need", "needn", "needn't", "needs", "neither", "never", "nevertheless", "new", "next", "ng", "ni", "nine", "ninety", "nj", "nl", "nn", "no", "nobody", "non", "none", "nonetheless", "noone", "nor", "normally", "nos", "not", "noted", "nothing", "novel", "now", "nowhere", "nr", "ns", "nt", "ny", "o", "oa", "ob", "obtain", "obtained", "obviously", "oc", "od", "of", "off", "often", "og", "oh", "oi", "oj", "ok", "okay", "ol", "old", "om", "omitted", "on", "once", "one", "ones", "only", "onto", "oo", "op", "oq", "or", "ord", "os", "ot", "other", "others", "otherwise", "ou", "ought", "our", "ours", "ourselves", "out", "outside", "over", "overall", "ow", "owing", "own", "ox", "oz", "p", "p1", "p2", "p3", "page", "pagecount", "pages", "par", "part", "particular", "particularly", "pas", "past", "pc", "pd", "pe", "per", "perhaps", "pf", "ph", "pi", "pj", "pk", "pl", "placed", "please", "plus", "pm", "pn", "po", "poorly", "possible", "possibly", "potentially", "pp", "pq", "pr", "predominantly", "present", "presumably", "previously", "primarily", "probably", "promptly", "proud", "provides", "ps", "pt", "pu", "put", "py", "q", "qj", "qu", "que", "quickly", "quite", "qv", "r", "r2", "ra", "ran", "rather", "rc", "rd", "re", "readily", "really", "reasonably", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related", "relatively", "research", "research-articl", "respectively", "resulted", "resulting", "results", "rf", "rh", "ri", "right", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "run", "rv", "ry", "s", "s2", "sa", "said", "same", "saw", "say", "saying", "says", "sc", "sd", "se", "sec", "second", "secondly", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self", "selves", "sensible", "sent", "serious", "seriously", "seven", "several", "sf", "shall", "shan", "shan't", "she", "shed", "she'd", "she'll", "shes", "she's", "should", "shouldn", "shouldn't", "should've", "show", "showed", "shown", "showns", "shows", "si", "side", "significant", "significantly", "similar", "similarly", "since", "sincere", "six", "sixty", "sj", "sl", "slightly", "sm", "sn", "so", "some", "somebody", "somehow", "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry", "sp", "specifically", "specified", "specify", "specifying", "sq", "sr", "ss", "st", "still", "stop", "strongly", "sub", "substantially", "successfully", "such", "sufficiently", "suggest", "sup", "sure", "sy", "system", "sz", "t", "t1", "t2", "t3", "take", "taken", "taking", "tb", "tc", "td", "te", "tell", "ten", "tends", "tf", "th", "than", "thank", "thanks", "thanx", "that", "that'll", "thats", "that's", "that've", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "thered", "therefore", "therein", "there'll", "thereof", "therere", "theres", "there's", "thereto", "thereupon", "there've", "these", "they", "theyd", "they'd", "they'll", "theyre", "they're", "they've", "thickv", "thin", "think", "third", "this", "thorough", "thoroughly", "those", "thou", "though", "thoughh", "thousand", "three", "throug", "through", "throughout", "thru", "thus", "ti", "til", "tip", "tj", "tl", "tm", "tn", "to", "together", "too", "took", "top", "toward", "towards", "tp", "tq", "tr", "tried", "tries", "truly", "try", "trying", "ts", "t's", "tt", "tv", "twelve", "twenty", "twice", "two", "tx", "u", "u201d", "ue", "ui", "uj", "uk", "um", "un", "under", "unfortunately", "unless", "unlike", "unlikely", "until", "unto", "uo", "up", "upon", "ups", "ur", "us", "use", "used", "useful", "usefully", "usefulness", "uses", "using", "usually", "ut", "v", "va", "value", "various", "vd", "ve", "ve", "very", "via", "viz", "vj", "vo", "vol", "vols", "volumtype", "vq", "vs", "vt", "vu", "w", "wa", "want", "wants", "was", "wasn", "wasnt", "wasn't", "way", "we", "wed", "we'd", "welcome", "well", "we'll", "well-b", "went", "were", "we're", "weren", "werent", "weren't", "we've", "what", "whatever", "what'll", "whats", "what's", "when", "whence", "whenever", "when's", "where", "whereafter", "whereas", "whereby", "wherein", "wheres", "where's", "whereupon", "wherever", "whether", "which", "while", "whim", "whither", "who", "whod", "whoever", "whole", "who'll", "whom", "whomever", "whos", "who's", "whose", "why", "why's", "wi", "widely", "will", "willing", "wish", "with", "within", "without", "wo", "won", "wonder", "wont", "won't", "words", "world", "would", "wouldn", "wouldnt", "wouldn't", "www", "x", "x1", "x2", "x3", "xf", "xi", "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y", "y2", "yes", "yet", "yj", "yl", "you", "youd", "you'd", "you'll", "your", "youre", "you're", "yours", "yourself", "yourselves", "you've", "yr", "ys", "yt", "z", "zero", "zi", "zz",]
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import pyttsx3
import speech_recognition as sr
import gtts
from moviepy.editor import *
import pytesseract
import cv2
from fpdf import FPDF
import os
import random

#text_speech=pyttsx3.init()

import pandas as pd
def haterd(text):
    #text_speech=pyttsx3.init()
    text1=text

    df = pd.read_csv('toxicity_en.csv')

    df.is_toxic.replace("Toxic", 1, inplace=True)
    df.is_toxic.replace("Not Toxic", 2, inplace=True)

    def data_processing(text):
        text= text.lower()
        text = re.sub('<br />', '', text)
        text = re.sub(r"https\S+|www\S+|http\S+", '', text, flags = re.MULTILINE)
        text = re.sub(r'\@w+|\#', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text_tokens = word_tokenize(text)
        filtered_text = [w for w in text_tokens if not w in stop_words]
        return " ".join(filtered_text)

    df.text = df['text'].apply(data_processing)

    duplicated_count = df.duplicated().sum()
    print("Number of duplicate entries: ", duplicated_count)

    df = df.drop_duplicates('text')

    stemmer = PorterStemmer()
    def stemming(data):
        text = [stemmer.stem(word) for word in data]
        return data

    df.text = df['text'].apply(lambda x: stemming(x))

    def no_of_words(text):
        words= text.split()
        word_count = len(words)
        return word_count

    df['word count'] = df['text'].apply(no_of_words)
    df.head()

    toxic =  df[df.is_toxic == 1]
    toxic.head()

    non_toxic =  df[df.is_toxic == 2]
    non_toxic.head()

    X = df['text']
    Y = df['is_toxic']

    vect = TfidfVectorizer()
    X = vect.fit_transform(df['text'])

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    print("Size of x_train: ", (x_train.shape))
    print("Size of y_train: ", (y_train.shape))
    print("Size of x_test: ", (x_test.shape))
    print("Size of y_test: ", (y_test.shape))

    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import warnings
    warnings.filterwarnings('ignore')

    logreg = LogisticRegression()
    logreg.fit(x_train, y_train)
    logreg_pred = logreg.predict(x_test)
    logreg_acc = accuracy_score(logreg_pred, y_test)
    print("Test accuracy of logistic Regression : {:.2f}%".format(logreg_acc*100))
    print(confusion_matrix(y_test, logreg_pred))
    print("\n")
    print(classification_report(y_test, logreg_pred))


    mnb = MultinomialNB()
    mnb.fit(x_train, y_train)
    mnb_pred = mnb.predict(x_test)
    mnb_acc = accuracy_score(mnb_pred, y_test)
    print("Test accuracy of Multinomial Naive Bayes: {:.2f}%".format(mnb_acc*100))
    print(confusion_matrix(y_test, mnb_pred))
    print("\n")
    print(classification_report(y_test, mnb_pred))
    


    svc = LinearSVC()
    svc.fit(x_train, y_train)
    svc_pred = svc.predict(x_test)
    svc_acc = accuracy_score(svc_pred, y_test)
    print("Test accuracy Support Vector Classification : {:.2f}%".format(svc_acc*100))
    print(confusion_matrix(y_test, svc_pred))
    print("\n")
    print(classification_report(y_test, svc_pred))


    


    x=text1
    result=""
    my_text = pd.Series([x])
    X = vect.transform(my_text)
    pred = logreg.predict(X)
    if pred[0] == 1:       
        result=" It's toxic speech"       
    else:
        result=" It's not toxic speech"
    """    
    while True:  
        engine = pyttsx3.init()  
        rate=engine.getProperty("rate")
        engine.setProperty("rate",120)
        engine.say("your sentence is " + x + result )
        engine.runAndWait() 
        engine = None 
        break  """
    return result 

def voice():
    r= sr.Recognizer()
    with sr.Microphone() as source:
        print("listening.....")
        audio= r.listen(source)
        try:
            text= r.recognize_google(audio)
            print("You said : {} ".format(text))
        except:
            print("Sorry could not recognize your voice")
            return None, None

    result=haterd(text)         
    return result,text


def audfie(file_path):
    r=sr.Recognizer()
    text=""
    with sr.AudioFile(file_path) as source:
        audio=r.record(source)
        try:
            text=r.recognize_google(audio)
            print("working on....")
            print(text)
        except:
            audfie(file_path)

    result=haterd(text)
    return result,text    


def imgtotext(file_path):
    pytesseract.pytesseract.tesseract_cmd=r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    img=cv2.imread(file_path)
    text=pytesseract.image_to_string(img)
    if text=="":
        result=""
        return result,text
    else:
        text = text.replace("\n", " ")
        result=haterd(text)
        return result,text

def pdfcreate(text1,text2):
    page=FPDF()
    page.add_page()
    download_folder = os.path.expanduser("~") + "/Downloads/"
    

    page.set_draw_color(0, 0, 0)  # RGB color for the border 
    page.set_line_width(1)  # Width of the border line in points
    page.rect(10, 10, page.w - 20, page.h - 20)  # Set the border around the page

    page.set_font("Arial","B",23)
    page.cell(105,20,"Hate Speech Detection")

    #  position for sentence 1 and text1
    sentence1_x = 20
    sentence1_y = 40
    text1_x = 20
    text1_y = 50

    # position for sentence 2 and text2
    sentence2_x = 20
    sentence2_y = text1_y + 20
    text2_x = 20
    text2_y = sentence2_y + 10

    page.set_font("Arial","B",18)
    # Draw sentence 1 and text1
    page.set_xy(sentence1_x, sentence1_y)
    page.cell(0, 0, "Sentence :", align="L")


    page.set_font("Arial","B",15)
    page.set_xy(text1_x, text1_y)
    page.multi_cell(0, 10, text1, align="L")

    gap = 50  # Adjust the value for the desired gap
    page.set_xy(sentence2_x, text1_y + gap)
    

    # Draw sentence 2 and text2
    page.set_font("Arial","B",18)
    page.set_xy(sentence2_x, page.get_y() + gap)
    page.cell(0, 0, "Result :", align="L")

    page.set_font("Arial","B",15)
    page.set_xy(text2_x, page.get_y() + 10) 
    page.multi_cell(0, 10, text2, align="L")
    filename="report"+str(random.randint(1,100))+".pdf"
    page.output(download_folder + filename, "F")

def ftotext(file_path):
    text=""
    x=open(file_path)
    for line in x:
        text+=line
    text = text.replace("\n", " ")
    result=haterd(text)
    return result,text 
    
    


