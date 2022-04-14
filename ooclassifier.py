#--------------------------------------------
#   Assignment #1: OO Classifier
#-------------------------------------------- 
import sys
import copy     # for deepcopy()
import string
import collections
from collections import Counter

Debug = False   # Sometimes, print for debugging.  Overridable on command line.
InputFilename = "file.input.txt"
TargetWords = [
        'outside', 'today', 'weather', 'raining', 'nice', 'rain', 'snow',
        'day', 'winter', 'cold', 'warm', 'snowing', 'out', 'hope', 'boots',
        'sunny', 'windy', 'coming', 'perfect', 'need', 'sun', 'on', 'was',
        '-40', 'jackets', 'wish', 'fog', 'pretty', 'summer'
        ]


def open_file(filename=InputFilename):
    try:
        f = open(filename, "r")
        return(f)
    except FileNotFoundError:
        # FileNotFoundError is subclass of OSError
        if Debug:
            print("File Not Found")
        return(sys.stdin)
    except OSError:
        if Debug:
            print("Other OS Error")
        return(sys.stdin)


def safe_input(f=None, prompt=""):
    try:
        # Case:  Stdin
        if f is sys.stdin or f is None:
            line = input(prompt)
        # Case:  From file
        else:
            assert not (f is None)
            assert (f is not None)
            line = f.readline()
            if Debug:
                print("readline: ", line, end='')
            if line == "":  # Check EOF before strip()
                if Debug:
                    print("EOF")
                return("", False)
        return(line.strip(), True)
    except EOFError:
        return("", False)


class C274:
    def __init__(self):
        self.type = str(self.__class__)
        return

    def __str__(self):
        return(self.type)

    def __repr__(self):
        s = "<%d> %s" % (id(self), self.type)
        return(s)

 
class ClassifyByTarget(C274):
    def __init__(self, lw=[]):
        super().__init__() # Call superclass
        self.type = str(self.__class__)
        self.allWords = 0
        self.theCount = 0
        self.nonTarget = []
        self.set_target_words(lw)
        self.initTF()
        return

    def initTF(self):
        self.TP = 0
        self.FP = 0
        self.TN = 0
        self.FN = 0
        return

    # FIXME:  Incomplete.  Finish get_TF() and other getters/setters.
    def get_TF(self):
        return(self.TP, self.FP, self.TN, self.FN)

    # TODO: Could use Use Python properties
    #     https://www.python-course.eu/python3_properties.php
    def set_target_words(self, lw):
        # Could also do self.targetWords = lw.copy().  Thanks, TA Jason Cannon
        self.targetWords = copy.deepcopy(lw)
        return

    def get_target_words(self):
        return(self.targetWords)

    def get_allWords(self):
        return(self.allWords)

    def incr_allWords(self):
        self.allWords += 1
        return

    def get_theCount(self):
        return(self.theCount)

    def incr_theCount(self):
        self.theCount += 1
        return

    def get_nonTarget(self):
        return(self.nonTarget)

    def add_nonTarget(self, w):
        self.nonTarget.append(w)
        return

    def print_config(self,printSorted=True):
        print("-------- Print Config --------")
        ln = len(self.get_target_words())
        print("TargetWords (%d): " % ln, end='')
        if printSorted:
            print(sorted(self.get_target_words()))
        else:
            print(self.get_target_words())
        return

    def print_run_info(self,printSorted=True):
        print("-------- Print Run Info --------")
        print("All words:%3s. " % self.get_allWords(), end='')
        print(" Target words:%3s" % self.get_theCount())
        print("Non-Target words (%d): " % len(self.get_nonTarget()), end='')
        if printSorted:
            print(sorted(self.get_nonTarget()))
        else:
            print(self.get_nonTarget())
        return

    def print_confusion_matrix(self, targetLabel, doKey=False, tag=""):
        assert (self.TP + self.TP + self.FP + self.TN) > 0
        print(tag+"-------- Confusion Matrix --------")
        print(tag+"%10s | %13s" % ('Predict', 'Label'))
        print(tag+"-----------+----------------------")
        print(tag+"%10s | %10s %10s" % (' ', targetLabel, 'not'))
        if doKey:
            print(tag+"%10s | %10s %10s" % ('', 'TP   ', 'FP   '))
        print(tag+"%10s | %10d %10d" % (targetLabel, self.TP, self.FP))
        if doKey:
            print(tag+"%10s | %10s %10s" % ('', 'FN   ', 'TN   '))
        print(tag+"%10s | %10d %10d" % ('not', self.FN, self.TN))
        return

    def eval_training_set(self, tset, targetLabel, lines=True):
        print("-------- Evaluate Training Set --------")
        self.initTF()
        # zip is good for parallel arrays and iteration
        z = zip(tset.get_instances(), tset.get_lines())
        for ti, w in z:
            lb = ti.get_label()
            cl = ti.get_class()
            if lb == targetLabel:
                if cl:
                    self.TP += 1
                    outcome = "TP"
                else:
                    self.FN += 1
                    outcome = "FN"
            else:
                if cl:
                    self.FP += 1
                    outcome = "FP"
                else:
                    self.TN += 1
                    outcome = "TN"
            explain = ti.get_explain()
            # Format nice output
            if lines:
                w = ' '.join(w.split())
            else:
                w = ' '.join(ti.get_words())
                w = lb + " " + w

            # TW = testing bag of words words (kinda arbitrary)
            print("TW %s: ( %10s) %s" % (outcome, explain, w))
            if Debug:
                print("-->", ti.get_words())
        self.print_confusion_matrix(targetLabel)
        return

    def classify_by_words(self, ti, update=False, tlabel="last"):
        inClass = False
        evidence = ''
        lw = ti.get_words()
        for w in lw:
            if update:
                self.incr_allWords()
            if w in self.get_target_words():    # FIXME Write predicate
                inClass = True
                if update:
                    self.incr_theCount()
                if evidence == '':
                    evidence = w            # FIXME Use first word, but change
            elif w != '':
                if update and (w not in self.get_nonTarget()):
                    self.add_nonTarget(w)
        if evidence == '':
            evidence = '#negative'
        if update:
            ti.set_class(inClass, tlabel, evidence)
        return(inClass, evidence)

    # Could use a decorator, but not now
    def classify(self, ti, update=False, tlabel="last"):
        cl, e = self.classify_by_words(ti, update, tlabel)
        return(cl, e)

    def classify_all(self, ts, update=True, tlabel="classify_all"):
        for ti in ts.get_instances():
            cl, e = self.classify(ti, update=update, tlabel=tlabel)
        return


class TrainingInstance(C274):
    def __init__(self):
        super().__init__() # Call superclass
        # self.type = str(self.__class__)
        self.inst = dict()
        # FIXME:  Get rid of dict, and use attributes
        self.inst["label"] = "N/A"      # Class, given by oracle
        self.inst["words"] = []         # Bag of words
        self.inst["class"] = ""         # Class, by classifier
        self.inst["explain"] = ""       # Explanation for classification
        self.inst["experiments"] = dict()   # Previous classifier runs
        return

    def get_label(self):
        return(self.inst["label"])

    def get_words(self):
        return(self.inst["words"])

    def set_class(self, theClass, tlabel="last", explain=""):
        # tlabel = tag label
        self.inst["class"] = theClass
        self.inst["experiments"][tlabel] = theClass
        self.inst["explain"] = explain
        return

    def get_class_by_tag(self, tlabel):             # tlabel = tag label
        cl = self.inst["experiments"].get(tlabel)
        if cl is None:
            return("N/A")
        else:
            return(cl)

    def get_explain(self):
        cl = self.inst.get("explain")
        if cl is None:
            return("N/A")
        else:
            return(cl)

    def get_class(self):
        return self.inst["class"]

    def process_input_line(
                self, line, run=None,
                tlabel="read", inclLabel=False
            ):
        for w in line.split():
            if w[0] == "#":
                self.inst["label"] = w
                if inclLabel:
                    self.inst["words"].append(w)
            else:
                self.inst["words"].append(w)

        if not (run is None):
            cl, e = run.classify(self, update=True, tlabel=tlabel)
        return(self)

    #preprocess_words is called by preprocess, in TrainingSet. This 
    #method is responsible for preprocessing the text. Details to follow.
    def preprocess_words(self, mode=''):
        #The input text is converted to all letters to lowercase.
        def lowercase(bag_of_words):
            lower = str(bag_of_words)
            lower = lower.lower()
            return(lower)

        #Any punctuations within the text are removed using the import string and string.punctuation repository.
        def no_punctuation(lower):
            punc_free = "".join([i for i in lower if i not in string.punctuation])
            return(punc_free)

        #Numbers are removed, unless the word consists entirely of numbers.
        def no_numbers(punc_free):
            numb_free = []
            for word in punc_free.split():
                if not any(c.isalpha() for c in word):
                    word = "".join([char for char in word])
                    numb_free.append(word)
                elif any(c.isdigit() for c in word):
                    word = "".join([char for char in word if not char.isdigit()])
                    numb_free.append(word)
                else:
                    numb_free.append(word)
            numb_free = " ".join([word for word in numb_free])
            return(numb_free)

        #ALL numbers are removed if "keep-symbols" is present.
        def no_numbers_special(punc_free):
            numb_free = []
            for word in punc_free.split():
                if any(c.isdigit() for c in word):
                    word = "".join([char for char in word if not char.isdigit()])
                    numb_free.append(word)
                else:
                    numb_free.append(word)
            numb_free = " ".join([word for word in numb_free])
            return(numb_free)

        #All stop words are removed if they are present in stop_words. Otherwise, they are kept.
        def no_stop(numb_free):
            stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
            "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
            "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
            "themselves", "what", "which","who", "whom", "this", "that", "these", "those",
            "am", "is", "are", "was", "were", "be","been", "being", "have", "has", "had",
            "having", "do", "does", "did", "doing", "a", "an","the", "and", "but", "if",
            "or", "because", "as", "until", "while", "of", "at", "by", "for", "with",
            "about", "against", "between", "into", "through", "during", "before", "after",
            "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over",
            "under", "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "any", "both", "each", "few", "more", "most", "other",
            "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
            stop_free = []
            for word in numb_free.split():
                if word not in stop_words:
                    stop_free.append(word)
            #stop_free = " ".join([word for word in stop_free])
            return(stop_free)

        #Finally, the text that has been kept is printed to the terminal.
        def final(stop_free):
            final = stop_free
            self.inst["words"] = final
        
        #main allows for specific methods within preprocess_words to be
        #called based on the mode that is present, such as keep-digits,
        #keep-symbols, or keep-stops.
        def preprocess_main():
            bag_of_words = self.inst["words"]
            if mode == '':
                lower = lowercase(bag_of_words)
                punc_free = no_punctuation(lower)
                numb_free = no_numbers(punc_free)
                stop_free = no_stop(numb_free)
                final(stop_free)
            elif mode == "keep-symbols":
                lower = lowercase(bag_of_words)
                numb_free = no_numbers_special(lower)
                stop_free = no_stop(numb_free)
                final(stop_free) 
            elif mode == "keep-digits":
                lower = lowercase(bag_of_words)
                punc_free = no_punctuation(lower)
                stop_free = no_stop(punc_free)
                final(stop_free)
            elif mode == "keep-stops":
                lower = lowercase(bag_of_words)
                punc_free = no_punctuation(lower)
                numb_free = no_numbers(punc_free)
                final(numb_free) 
        #Call to preprocess_main() in order to initiate different modes
        preprocess_main()
          
class TrainingSet(C274):
    def __init__(self):
        super().__init__() # Call superclass
        # self.type = str(self.__class__)
        self.inObjList = []     # Unparsed lines, from training set
        self.inObjHash = []     # Parsed lines, in dictionary/hash
        self.variable = dict()  # NEW: Configuration/environment variables
        return

    def set_env_variable(self, k, v):
        self.variable[k] = v
        return

    def get_env_variable(self, k):
        if k in self.variable:
            return(self.variable[k])
        else:
            return ""

    def inspect_comment(self, line):
        if len(line) > 1 and line[1] != ' ':      # Might be variable
            v = line.split(maxsplit=1)
            self.set_env_variable(v[0][1:], v[1])
        return

    def get_instances(self):
        return(self.inObjHash)      # FIXME Should protect this more

    def get_lines(self):
        return(self.inObjList)      # FIXME Should protect this more

    def print_training_set(self):
        print("-------- Print Training Set --------")
        z = zip(self.inObjHash, self.inObjList)
        for ti, w in z:
            lb = ti.get_label()
            cl = ti.get_class_by_tag("last")     # Not used
            explain = ti.get_explain()
            print("( %s) (%s) %s" % (lb, explain, w))
            if Debug:
                print("-->", ti.get_words())
        return

    def process_input_stream(self, inFile, run=None):
        assert not (inFile is None), "Assume valid file object"
        cFlag = True
        while cFlag:
            line, cFlag = safe_input(inFile)
            if not cFlag:
                break
            assert cFlag, "Assume valid input hereafter"

            if len(line) == 0:   # Blank line.  Skip it.
                continue

            # Check for comments *and* environment variables
            if line[0] == '%':  # Comments must start with % and variables
                self.inspect_comment(line)
                continue

            # Save the training data input, by line
            self.inObjList.append(line)
            # Save the training data input, after parsing
            ti = TrainingInstance()
            ti.process_input_line(line, run=run)
            self.inObjHash.append(ti)
        return

    #Call to preprocess_words in TrainingInstant class.
    #Iterates through inObjHash and preprocessing each iteration
    def preprocess(self, mode=''):
        for i in self.inObjHash:
            i.preprocess_words(mode)

    #Method that returns a list of num objects of class TrainingSet.
    #Each partition/fold for each object contains a deepcopy of all
    #attributes for that instance. Round robin strategy is used,
    #containing 3 discrete partitions and, depending on the objects
    #of class TrainingInstance, can return a non-3 partition (e.g. 7).
    def return_nfolds(self, num=3):
        nfolds = []
        #Based on the training instance, both inObjHash and inObjList
        #are appending/revised.
        for i in range(num):
            train = TrainingSet()
            count = i
            while count < len(self.inObjHash):
                deeper_hash = copy.deepcopy(self.inObjHash[count])
                train.inObjHash.append(deeper_hash)
                #train.inObjHash.append(copy.deepcopy(self.inObjHash[count]))
                deeper_list = copy.deepcopy(self.inObjList[count])
                train.inObjList.append(deeper_list)
                #train.inObjList.append(copy.deepcopy(self.inObjList[count]))
                count += num
            nfolds.append(train)
        return(nfolds)

    #Create a deep (not shallow) copy of object of class TrainingSet
    #and return the deep copy.
    def copy(self):
        copy_deeper = copy.deepcopy(self)
        return(copy_deeper)

    #Deep-copy of all training instances of tset are appended to the
    #objects of class TrainingSet, inObjHash & inObjList.
    def add_training(self, tset):
        
        for i in range(len(tset.inObjHash)):
            deeper_hash = copy.deepcopy(tset.inObjHash[i])
            self.inObjHash.append(deeper_hash)
            deeper_list = copy.deepcopy(tset.inObjList[i])
            self.inObjList.append(deeper_list)

        return


class ClassifyByTopN(ClassifyByTarget):
    #Method in subclass of ClassifyByTarget creates a word dictionary
    #and then adds words from inObjList, creating a frequency count.
    def target_top_n(self, tset, num=5, label=''):
        word_count = {}
        
        for line in tset.inObjList:
                if label in line.split():
                    line = line.split()
                    line.remove(label)
                    for word in line:
                        if word not in word_count:
                            word_count[word] = 1
                        else:
                            word_count[word] += 1
        #The dictionary, word_count, is then used to find the top 5
        #most frequent words from the created dictionary. This value
        #is applied to replace the already-existent targetWords.                
        words = []
        freq = list(word_count.values())
        freq.sort(reverse = True)
        for k, v in word_count.items():
            if v in freq[0: num]:
                words.append(k)
            elif v == v + 1:
                words.append(k)

        self.targetWords = words

# Very basic test of functionality
def basemain():
    global Debug
    tset = TrainingSet()
    run1 = ClassifyByTarget(TargetWords)
    if Debug:
        print(run1)     # Just to show __str__
        lr = [run1]
        print(lr)       # Just to show __repr__

    argc = len(sys.argv)
    if argc == 1:   # Use stdin, or default filename
        inFile = open_file()
        assert not (inFile is None), "Assume valid file object"
        tset.process_input_stream(inFile, run1)
        inFile.close()
    else:
        for f in sys.argv[1:]:
            # Allow override of Debug from command line
            if f == "Debug":
                Debug = True
                continue
            if f == "NoDebug":
                Debug = False
                continue

            inFile = open_file(f)
            assert not (inFile is None), "Assume valid file object"
            tset.process_input_stream(inFile, run1)
            inFile.close()

    print("--------------------------------------------")
    plabel = tset.get_env_variable("pos-label")
    print("pos-label: ", plabel)
    print("NOTE: Not using any target words from the file itself")
    print("--------------------------------------------")

    if Debug:
        tset.print_training_set()
    run1.print_config()
    run1.print_run_info()
    run1.eval_training_set(tset, plabel)

    return


if __name__ == "__main__":
    basemain()
