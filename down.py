import gdown

# a file
# url = "https://drive.google.com/uc?id=1l_5RK28JRL19wpT22B-DY9We3TVXnnQQ"
output = "pretrain_model.zip"
# gdown.download(url, output, quiet=False)

# same as the above, but with the file ID
# https://drive.google.com/file/d/1Ly_3RR1CsYDafdvdfTG35NPIG-FLH-tz/view?usp=sharing
id = "1Ly_3RR1CsYDafdvdfTG35NPIG-FLH-tz"
gdown.download(id=id, output=output, quiet=False)

# # same as the above, and you can copy-and-paste a URL from Google Drive with fuzzy=True
# url = "https://drive.google.com/file/d/0B9P1L--7Wd2vNm9zMTJWOGxobkU/view?usp=sharing"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# # cached download with identity check via MD5
# md5 = "fa837a88f0c40c513d975104edf3da17"
# gdown.cached_download(url, output, md5=md5, postprocess=gdown.extractall)

# a folder
# url = "https://drive.google.com/drive/folders/1RlqGBMo45lTmWz9MUPTq"
# gdown.download_folder(url, quiet=True, use_cookies=False)

# same as the above, but with the folder ID
# id = "15uNXeRBIhVvZJIhL4yTw4IsStMhUaaxl"
# gdown.download_folder(id=id, quiet=True, use_cookies=False)