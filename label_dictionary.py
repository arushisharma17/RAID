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


    def convert_label(self, label):
        label_l = label.lower()
        check = label_l in self.label_types
        if not check:
            return label.upper()
        return self.label_types[label.lower()]