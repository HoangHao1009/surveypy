
def delete(question, delete_list, max_length=80):
    filter_condition = []
    current_line = ''
    for i in delete_list:
        new_line = current_line + f'{question} = {i} OR '
        if len(new_line) > max_length:
            filter_condition.append(current_line)
            current_line = ''
        current_line = new_line

        filter_condition.append(current_line)

        command = ' \n'.join(filter_condition).rstrip(' OR ')

        return f'''
SELECT IF NOT ({command}).
EXECUTE.
'''

def var_label(question, label):
    return f"VARIABLE LABELS {question} '{label}'."

def value_label(question, label_dict):
    label = ''
    for i, v in label_dict.items():
        label += f'{i} "{v}"\n'
    label = label.rstrip('\n')
    return f"VALUE LABELS {question} {label}."

def mrset(question, question_label, list_answer, type='md'):
    if type == 'md':
        gr = 'MDGROUP'
        cate_value = 'CATEGORYLABELS=COUNTEDVALUES VALUE=1'
    elif type == 'mc':
        gr = 'MCGROUP'
        cate_value = ''
    else:
        raise(f'Type {type} is not valid')
    return f'''
MRSETS /{gr} NAME=${question}
LABEL='{question_label}'
{cate_value}
VARIABLES={' '.join(list_answer)}
/DISPLAY NAME=[${question}].
'''

def ctab(cols, cacl_dict=dict, comparetest_type=["MEAN"], alpha=0.1):
    #comparetest_type=["PROP"]
    def table_code(cacl_dict):
        code = '/TABLE '
        cal_command = {
            'Count': '[C][COUNT F40.0, TOTALS[COUNT F40.0]]',
            'ColPct': '[C][COLPCT.COUNT PCT40.0, TOTALS[COUNT F40.0]]',
            'Mean': '[MEAN COMMA40.2]',
            'Std': '[STDDEV COMMA40.2]'
        }

        for question, cal_list in cacl_dict.items():
            for cal in cal_list:
                code += f'{question} {cal_command[cal]} + \n'
        return code.rstrip(' + \n')
    
    def by_code(cols):
        return 'BY ' + ' + '.join(cols)
    
    def compare_code(comparetest_type, alpha):
        code = '/COMPARETEST'
        for test in comparetest_type:
            code += f'''
TYPE={test} ALPHA={alpha} ADJUST=NONE ORIGIN=COLUMN INCLUDEMRSETS=YES
CATEGORIES=ALLVISIBLE MEANSVARIANCE=TESTEDCATS MERGE=YES STYLE=SIMPLE SHOWSIG=NO'''
        return code
    return f'''
CTABLES
{table_code(cacl_dict)}
{by_code(cols)}
/SLABELS POSITION=ROW VISIBLE=NO
/CATEGORIES VARIABLES=ALL
    EMPTY=INCLUDE TOTAL=YES POSITION=BEFORE
{compare_code(comparetest_type, alpha)}.
'''

def compute_topbottom(question, type = '1-5'):

    new_question = f'{question}TB'
    def take_command(type):
        if type == '1-5':
            if_command = f'''
IF ({question} = 1 OR {question} = 2) {new_question} = 1.
IF ({question} = 3) {new_question} = 2.
IF ({question} = 4 OR {question} = 5) {new_question} = 3.
    '''
            value_label_command = value_label(new_question, {1: 'Bottom 2 boxes', 2: 'Neutral', 3: 'Top 2 boxes'})
        elif type == '1-10':
            if_command = f'''
IF ({question} = 1 OR {question} = 2 OR {question} = 3 OR {question} = 4 OR {question} = 5) {new_question} = 1.
IF ({question} = 6 OR {question} = 7) {new_question} = 2.
IF ({question} = 8 OR {question} = 9 OR {question} = 10) {new_question} = 3.
    '''
            value_label_command = value_label(new_question, {1: 'Bottom 5 boxes', 2: 'Neutral', 3: 'Top 3 boxes'})
        else:
            print(f'{type} is not valid')
            if_command, value_label_command = None, None
            
        return if_command, value_label_command

    if_command, value_label_command = take_command(type)

    label = f'{question} - Top to Bottom Boxes'

    command =  f'''
COMPUTE {new_question} = 0.
{if_command}
{var_label(new_question, label)}
{value_label_command}
EXECUTE.
'''
    
    return new_question, command

def compute_scale(question):
    new_question = f'{question}S'
    label = f'{question}. Scale'
    command =  f'''
COMPUTE {new_question} = {question}.
VARIABLE LEVEL {new_question} (SCALE).
{var_label(new_question, label)}
EXECUTE.
'''
    return new_question, command

def export(folder_path):
    return f'''
OUTPUT EXPORT
/CONTENTS  EXPORT=VISIBLE  LAYERS=PRINTSETTING  MODELVIEWS=PRINTSETTING
/XLSX  DOCUMENTFILE='{folder_path}'
    OPERATION=CREATEFILE
    LOCATION=LASTCOLUMN  NOTESCAPTIONS=NO.
'''

def compute_new_sa(sa_question_obj, compute_dict, label_dict, index):
    new_question = f'{sa_question_obj}_RECODE{index}'

    condition = ''

    for new, old_list in compute_dict.items():
        old_list = [str(i) for i in old_list]
        condition += f"({', '.join(old_list)} = {new})"
    command = f'''
RECODE {sa_question_obj.code} {condition} INTO {new_question}.
EXECUTE.
{var_label(new_question, f'RECODE - {sa_question_obj.text}')}
{value_label(new_question, label_dict)}
'''
    return new_question, command

def compute_new_ma(ma_question_obj, condition, label):
    index = len(ma_question_obj.sub_codes) + 1
    new_question = f'{ma_question_obj.code}A{index}'

    command =  f'''
COMPUTE {new_question} = 0.
IF ({condition}) {new_question} = 1.
EXECUTE.
{var_label(new_question, label)}
{value_label(new_question, {1: label})}
{mrset(ma_question_obj.code, ma_question_obj.text, ma_question_obj.sub_codes)}
'''
    return command
        
