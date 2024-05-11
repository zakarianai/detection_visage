import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import ntpath
import pyautogui

email = 'zeko1616naim@gmail.com'
password = 'zakaria1212'
send_to_email = 'zakarianaim56@gmail.com'
subject = 'detection'
message = 'detection'
file_location = 'students.txt'
msg = MIMEMultipart()
msg['From'] = email
msg['To'] = send_to_email
msg['Subject'] = subject
body = message
msg.attach(MIMEText(body, 'plain'))
filename = ntpath.basename(file_location)
attachment = open(file_location, "rt")
part = MIMEBase('application', 'octet-stream')
part.set_payload((attachment).read())
encoders.encode_base64(part)
part.add_header('Content-Disposition', "attachment; filename=%s" % filename)
msg.attach(part)
server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
server.starttls()
server.login(email, password)
text = msg.as_string()
server.sendmail(email, send_to_email, text)
server.quit()
