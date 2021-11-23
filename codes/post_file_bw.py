import ftplib
session = ftplib.FTP('brianmwalsh.com','USERNAME','PASSWORD')
file = open('/Users/bmwalsh/Documents/Research/DXL/SW_ACE.png','rb')                  # file to send
session.storbinary('STOR SW_ACE.png', file)     # send the file
file.close()                                    # close file and FTP
session.quit()
