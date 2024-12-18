import os
import re
import unittest
from raid.generate_files import TokenLabelFilesGenerator
from raid.label_dictionary import LabelDictionary


class RAIDTokenFunctions(unittest.TestCase):
    def confirm_equivalence(self, text, array_string):
        file_path = "input/test_file.txt"

        try:
            if not isinstance(text, bytes):
                text = re.sub(r'[^\x00-\x7F]+', '', text)
                text = bytes(text, 'utf-8')

            with open(file_path, "wb") as f:
                f.write(text)

            g = TokenLabelFilesGenerator()
            g.generate_in_label_bio_files('input/test_file.txt', 'java', 'program')

            with open(file_path, "r") as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                line = line.strip().replace(' ', '')
                self.assertTrue(line == array_string[i], 'Lines are not equal, ' + line + ' vs ' + array_string[i])

        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
            file_name = 'output/' + os.path.basename(file_path).split('.')[0]
            if os.path.exists(file_name + '.in'):
                os.remove(file_name + '.in')
            if os.path.exists(file_name + '.label'):
                os.remove(file_name + '.label')
            if os.path.exists(file_name + '.bio'):
                os.remove(file_name + '.bio')
            if os.path.exists(file_name + '.csv'):
                os.remove(file_name + '.csv')

    def test_multiline_code(self):
        bytestring = b'''for (StackTraceElement myStackElement : getStackTrace()) {
                b.append("\tat ").append(myStackElement).append('
                ');
                b.append("  ComposedException ").append(i).append(" :
                ");'''
        array_string = ['for(StackTraceElementmyStackElement:getStackTrace()){',
                        'b.append("\tat").append(myStackElement).append(\'',
                        "');",
                        'b.append("ComposedException").append(i).append(":',
                        '");',
                        '']

        self.confirm_equivalence(bytestring, array_string)


    def test_removing_invalid_characters(self):
        string = "builder\n.append(\"\n┌─for publisher \");"
        array_string = ['builder',
                        '.append("',
                        'forpublisher");',
                        '']

        self.confirm_equivalence(string, array_string)

    def test_escape_char(self):
        bytestring = b'''final String[] memberValues = value.split("\\|");
                 if (c == '\\') {}'''
        array_string = ['finalString[]memberValues=value.split("\\|");',
                        "if(c=='\\'){}",
                        '']
        self.confirm_equivalence(bytestring, array_string)


    def test_multiline_comment(self):
        bytestring = b'''/**
                      * {@inheritDoc}
                      */'''

        array_string = ['/**', '*{@inheritDoc}', '*/', '']

        self.confirm_equivalence(bytestring, array_string)


    def test_try_catch_with_single_comments(self):
        bytestring = b'''final String databaseName = database.getName();
                         try {
                            if (this.parsedStatement.modeFull) {
                            return replaceCluster(dManager, database, dManager.getServerInstance(), databaseName, this.parsedStatement.clusterName.getStringValue());
                          }
                          // else {
                          // int merged = 0;
                          // return String.format("Merged %d records", merged);
                          // }
                        } catch (Exception e) {
                          throw OException.wrapException(new OCommandExecutionException("Cannot execute synchronization of cluster"), e);
                        }
                        return "Mode not supported";
                    }'''

        array_string = ['finalStringdatabaseName=database.getName();',
                        'try{',
                        'if(this.parsedStatement.modeFull){',
                        'returnreplaceCluster(dManager,database,dManager.getServerInstance(),databaseName,this.parsedStatement.clusterName.getStringValue());',
                        '}',
                        '//else{',
                        '//intmerged=0;',
                        '//returnString.format("Merged%drecords",merged);',
                        '//}',
                        '}catch(Exceptione){',
                        'throwOException.wrapException(newOCommandExecutionException("Cannotexecutesynchronizationofcluster"),e);',
                        '}',
                        'return"Modenotsupported";',
                        '}',
                        '']

        self.confirm_equivalence(bytestring, array_string)


    def test_unusual_newlines(self):
        bytestring = b'''ODistributedServerLog
          .info( "Distributed (*=current @=lockmgr[%s]):
%s",
              getLockManagerServer(), ODistributedOutput.formatServerStatus(this, cfg));


");
        ncmlTA.appendLine( controller.getNcML());
        ncmlTA.gotoTop();
        ncmlWindow.show();
      }
    };
    BAMutil.setActionProperties( showNcMLAction, null, "Show NcML...", false, 'X', -1);

      // List           lines   = StringUtil.split(content, "
", false);
      String[] lines = content.split("
");
                    }'''
        array_string = ['ODistributedServerLog',
                        '.info("Distributed(*=current@=lockmgr[%s]):',
                        '%s",',
                        'getLockManagerServer(),ODistributedOutput.formatServerStatus(this,cfg));',
                        '',
                        '',
                        '");',
                        'ncmlTA.appendLine(controller.getNcML());',
                        'ncmlTA.gotoTop();',
                        'ncmlWindow.show();',
                        '}',
                        '};',
                        'BAMutil.setActionProperties(showNcMLAction,null,"ShowNcML...",false,\'X\',-1);',
                        '',
                        '//Listlines=StringUtil.split(content,"',
                        '",false);',
                        'String[]lines=content.split("',
                        '");',
                        '}',
                        '']
        self.confirm_equivalence(bytestring, array_string)


    def test_add_mixed_numbers(self):
        bytestring = b'''private static boolean mappedToNothing(final char ch) {
        return ch == '\u00AD'
                || ch == '\u034F'
                || ch == '\u1806'
                || ch == '\u180B'
                || ch == '\u180C'
                || ch == '\u180D'
                || ch == '\u200B'
                || ch == '\u200C'
                || ch == '\u200D'
                || ch == '\u2060'
                || '\uFE00' <= ch && ch <= '\uFE0F'
                || ch == '\uFEFF';
    }'''
        array_string = ["privatestaticbooleanmappedToNothing(finalcharch){",
                        "returnch=='\u00AD'",
                        "||ch=='\u034F'",
                        "||ch=='\u1806'",
                        "||ch=='\u180B'",
                        "||ch=='\u180C'",
                        "||ch=='\u180D'",
                        "||ch=='\u200B'",
                        "||ch=='\u200C'",
                        "||ch=='\u200D'",
                        "||ch=='\u2060'",
                        "||'\uFE00'<=ch&&ch<='\uFE0F'"
                        "||ch=='\uFEFF';"
                        "}"
                        ""]
        self.confirm_equivalence(bytestring, array_string)


    def test_add_mixed_numbers(self):
        bytestring = b'''if (true) {
                Map error_headers = new HashMap();
                error_headers.put( "message:", "authorization refused");
                error_headers.put( "type:", "send");
                error_headers.put( "channel:", destination);
                y.error( error_headers, "The message:
    -----
    "+b+
                    "
    -----
    Authentication token refused for this channel");
            } else
            { System.out.println("Test"); }'''
        array_string = ['if(true){',
                        'Maperror_headers=newHashMap();',
                        'error_headers.put("message:","authorizationrefused");',
                        'error_headers.put("type:","send");',
                        'error_headers.put("channel:",destination);',
                        'y.error(error_headers,"Themessage:',
                        '-----',
                        '"+b+',
                        '"',
                        '-----',
                        'Authenticationtokenrefusedforthischannel");',
                        '}else',
                        '{System.out.println("Test");}',
                        '']
        self.confirm_equivalence(bytestring, array_string)


if __name__ == '__main__':
    unittest.main()
    