import re
import os
def remove_comments(code):
    # 使用正则表达式删除以分号（;）开头的注释行
    code = re.sub(r';.*$', '', code, flags=re.MULTILINE)
    return code

def remove_debug_info(code):
    # 使用正则表达式删除形如"!insn.addr"的标记
    code = re.sub(r',\s!.*$', '', code, flags=re.MULTILINE)
    return code
def remove_useless_info(code):

    code = re.sub(r'#[0-9]*', '', code)
    code = re.sub(r'unnamed_addr', '', code)
    #code = re.sub(r'dec_label_.*$','1', code,flags=re.MULTILINE)
    return code
def normalize_label(code):
    a = 1
    dic={}
    regex=re.compile('; <label>:[0-9]*')
    match=regex.findall(code)
    if(match!=[]):
        for ele in match:
             num=re.findall(r'\d+',ele)
             for num1 in num:
               if str(num1) in dic:
                   code = re.sub("%" + num1 + ",", str(dic[num1]) + ",", code)
                   code = re.sub("label %" + num1, "label " + str(dic[num1]), code)
                   code = re.sub("; <label>:" + num1, str(dic[num1]), code)
               else:
                  dic[num1]=a
                  a=a+1
                  code=re.sub("; <label>:"+num1+":",str(dic[num1])+":",code)
                  code=re.sub("%"+num1+",",str(dic[num1])+",",code)
                  code = re.sub("label %"+num1, "label "+str(dic[num1]), code)

    return code


def normalize_code(code):
    # 删除调试信息
    code = remove_debug_info(code)
    #
    code=remove_useless_info(code)
    #
    code=normalize_label(code)
    # 去除注释
    code = remove_comments(code)
    return code



def process_allfiles(input_directory, output_directory):
    for filename in os.listdir(input_directory):
        if filename.endswith('.ll'):
            with open(os.path.join(input_directory, filename), 'r') as f:
                code = f.read()
            normalized_code = normalize_code(code)
            with open(os.path.join(output_directory, filename), 'w') as f:
                f.write(normalized_code)
                print(str(filename)+"    over")

# 标准化代码
input_directory=os.getcwd()
output_directory=os.getcwd()
process_allfiles(input_directory,output_directory)
