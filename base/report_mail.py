import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# 发件人邮箱
asender = ""
# 收件人邮箱
areceiver = ""
# 抄送人邮箱
acc = ""
# 邮件主题
asubject = "这是一封测试邮件"

# 发件人地址
from_addr = ""
# 邮箱授权码
password = ""

# 邮件设置
msg = MIMEMultipart()
msg['Subject'] = asubject
msg['to'] = areceiver
msg['Cc'] = acc
msg['from'] = ""

# 邮件正文
body = "你好，这是一份测试邮件"
msg.attach(MIMEText(body, 'plain', 'utf-8'))

# 添加附件
xlsxpart = MIMEApplication(open('./report.csv', 'rb').read())
xlsxpart.add_header('Content-Disposition', 'attachment', filename='report.csv')

msg.attach(xlsxpart)

# 设置邮箱服务器地址及端口
smtp_server = "smtp.163.com"
server = smtplib.SMTP(smtp_server, 25)
# 打印日志
server.set_debuglevel(1)

# 登录邮箱
server.login(from_addr, password)
# 发送邮件
server.sendmail(from_addr, areceiver.split(',') + acc.split(','), msg.as_string())

# 断开服务器连接
server.quit()
