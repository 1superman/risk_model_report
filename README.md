## 模型开发报告

### 报告内容主要包含以下几点<br>
<br>
<br>
a. 样本概况表。<br>
![image](https://github.com/1superman/model_report/tree/main/images/all_KS.png)
<br>
b. 效果概况。<br>
<br>
c. ROC曲线。<br>
<br>
d. KS曲线。<br>
<br>
e. lift曲线。<br>
<br>
f. 等频分箱。<br>
<br>
g. 分数分布。<br>
<br>
h. 累计分数分布。<br>
<br>
i. psi。<br>
<br>
j. 策略建议。<br>

### 输出形式<br>
word <br>
### 配置文件修改<br>
model_report/code中：client_id_name改为文件名（不需带后缀，默认pkl形式）
### 使用方式<br>
model_report/data/目录下存放切分文件：文件列包括：uid,triger_date,type,label,type<br>
model_report/data/score目录下存放结果文件：文件列在切分文件的基础上多一列p,并将时间列名改为：repayment_date<br>
配置完成后，执行"python model_report/code/Gen_Outside_Report_v3"<br>
### 注意事项<br>
文件model_report/data/report/标准模板.docx不可删除<br>
