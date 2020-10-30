import os
import importlib
import zipfile
import glob


def generate_outputs(pyfile_path, input_paths, output_dir):
    predict_py = pyfile_path + ".model_predict"
    define_py = pyfile_path + ".model_define"
    init_model = getattr(importlib.import_module(define_py), "init_model")
    predict = getattr(importlib.import_module(predict_py), "predict")

    model = init_model()
    for input_path in input_paths:
        predict(model, input_path, output_dir)


user_zip = 'submit.zip'  # 后台存储的选手上传的压缩包
f_size_MB = os.path.getsize(user_zip) / 1024.0 / 1024.0
print("[*] zip size: {}".format(f_size_MB))
if f_size_MB > 500:
    score = 0  # zip文件超过500得分为0
    print("[*] zip upper max: score 0.")

img_dir = "/nfs/users/huangfeifei/dataset/remote_sensing/test_multi_scale/image"
# 选手的代码会解压到当前路径的user文件夹下，依次必须按照上述说明去引入文件、函数、类等
with zipfile.ZipFile(user_zip, 'r') as f:
    f.extractall('./user')
img_paths = glob.glob(img_dir + '/' + '*')  # 后台存储的测试集图片路径
print("[*] image len: {}".format(len(img_paths)))
# 此处相当于使用了user.model_predict，因此选手需要将所有用到的文件直接打包为zip，而不是放到一个文件夹中再把文件夹压缩为zip
generate_outputs("user", img_paths, "./output")