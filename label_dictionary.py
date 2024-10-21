class LabelDictionary:
    def __init__(self):

        self.label_types = {"::": "DOUBLECOLON", "--": "DOUBLEMINUS", "++": "DOUBLEPLUS", "false": "BOOL", "true": "BOOL",
                      "modifier": "MODIFIER", "public": "MODIFIER", "basictype": "TYPE", "null": "IDENT", "keyword": "KEYWORD",
                       "identifier": "IDENT", "decimalinteger": "NUMBER", "decimalfloatingpoint": "NUMBER",
                      "string": "STRING", "string_fragment": "STRING",
                       "(": "LPAR", ")": "RPAR", "[": "LSQB", "]": "RSQB", ",": "COMMA", "?": "CONDITIONOP",
                      ";": "SEMI", "+": "PLUS", "-": "MINUS", "*": "STAR", "/": "SLASH", ".": "DOT", "=": "EQUAL", ":": "COLON",
                      "|": "VBAR", "&": "AMPER", "<": "LESS", ">": "GREATER", "%": "PERCENT", "{": "LBRACE", "}": "RBRACE",
                      "==": "EQEQUAL", "!=": "NOTEQUAL", "<=": "LESSEQUAL", ">=": "GREATEREQUAL", "~": "TILDE",
                      "^": "CIRCUMFLEX", "\"": "DQUOTES",
                      "<<": "LEFTSHIFT", ">>": "RIGHTSHIFT", "**": "DOUBLESTAR", "+=": "PLUSEUQAL", "-=": "MINEQUAL",
                      "*=": "STAREQUAL",
                      "/=": "SLASHEQUAL", "%=": "PERCENTEQUAL", "&=": "AMPEREQUAL", "|=": "VBAREQUAL", "^=": "CIRCUMFLEXEQUAL",
                       "<<=": "LEFTSHIFTEQUAL", ">>=": "RIGHTSHIFTEQUAL", "**=": "DOUBLESTAREQUAL", "//": "DOUBLESLASH",
                       "//=": "DOUBLESLASHEQUAL",
                       "@": "AT", "@=": "ATEQUAL", "->": "RARROW", "...": "ELLIPSIS", ":=": "COLONEQUAL", "&&": "AND",
                       "!": "NOT", "||": "OR"}

        self.keyword_types = {"abstract" : "keyword", "assert" : "keyword", "boolean" : "keyboard", "break" : "keyword",
                              "byte" : "keyword", "case" : "keyword", "catch" : "keyword", "char" : "keyword",
                              "class" : "keyword", "const" : "keyword", "continue" : "keyword", "default" : "keyword",
                              "do" : "keyword", "double" : "keyword", "else" : "keyword", "enum" : "keyword",
                              "extends" : "keyword", "final" : "keyword", "finally" : "keyword", "float" : "keyword",
                              "for" : "keyword", "goto" : "keyword", "if" : "keyword", "implements" : "keyword",
                              "import" : "keyword", "instanceof" : "keyword", "int" : "keyword",
                              "interface" : "keyword", "long" : "keyword", "native" : "keyword", "new" : "keyword",
                              "null" : "keyword", "package" : "keyword", "private" : "keyword", "protected" : "keyword",
                              "public" : "keyword", "return" : "keyword", "short" : "keyword", "static" : "keyword",
                              "strictfp" : "keyword", "super" : "keyword", "switch" : "keyword",
                              "synchronized" : "keyword", "this" : "keyword", "throw" : "keyword", "throws" : "keyword",
                              "transient" : "keyword", "try" : "keyword", "void" : "keyword", "volatile" : "keyword",
                              "while" : "keyword"}






    def convert_label(self, label):
        label_l = label.lower()
        check_for_keyword = label_l in self.keyword_types
        if check_for_keyword:
            label_l = self.keyword_types[label_l]
        check_for_label = label_l in self.label_types
        if not check_for_label:
            return label.upper()
        return self.label_types[label_l]