import datetime
import time
import json
from data_check import check
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, app, render_template, sessions, request, url_for, jsonify
from flask import flash
from sqlalchemy import ForeignKey, desc
from werkzeug.utils import redirect, secure_filename
import os
import pymysql
from algorithm import mlknn, onenn
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin, login_required, current_user, logout_user
from flask_login import LoginManager
from flask_login import login_user
from similarity import similar
from jaccard import compute_label
import pandas as pd
from Clustering2Analysis import Clustering2Analysis
from PR_system_sim import PrSystemSim
from tcmpr_test.tcmpr_test import TcmprModelTest

pymysql.install_as_MySQLdb()

app = Flask(__name__)

# 连接数据库
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://root:200189ymy@127.0.0.1:3306/dc'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = '23'
db = SQLAlchemy(app)

# 刻下症联想输入，读取全部症状
data = pd.read_excel('./datasets/allsymptoms.xlsx')
s = data['symptoms'].values
symptoms = []
for index in s:
    symptoms.append(str(index))


# 数据表创建

# 患者（编号，姓名，性别，年龄, 血型）
# 主键：编号
class Patient(db.Model):
    id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    name = db.Column(db.Text)
    sex = db.Column(db.Text)
    age = db.Column(db.Integer)
    blood = db.Column(db.String(10))


# 问诊信息（信息编号，主诉，既往史，证候，就诊详情，脉象，刻下症，西医诊断，舌象,患者，医生，划分数据ID，所属人群ID）
# 主键：信息编号
class InquiryInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    chief_complaint = db.Column(db.Text)
    past_history = db.Column(db.Text)
    zheng_hou = db.Column(db.Text)
    visit_type = db.Column(db.Integer)
    pulse_condition = db.Column(db.Text)
    syndrome = db.Column(db.Text)
    western_diagnostics = db.Column(db.Text)
    tongue_picture = db.Column(db.Text)
    cluster_id = db.Column(db.Integer)
    patient = db.Column(db.Integer, ForeignKey('patient.id'))
    doctor = db.Column(db.Integer, ForeignKey('user.id'))


# 问诊记录（记录号，患者，信息编号，医生，问诊时间）
# 主键：记录号，患者，信息编号，医生
class Inquiry(db.Model):
    id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    patient = db.Column(db.Integer, ForeignKey('patient.id'), primary_key=True)
    inquiry_info = db.Column(db.Integer, ForeignKey('inquiry_info.id'), primary_key=True)
    doctor = db.Column(db.Integer, ForeignKey('user.id'), primary_key=True)
    time = db.Column(db.DateTime)


# 处方（处方编号，组成药物，算法模型编号，问诊信息编号）
# 主键：处方编号
class Prescription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    drug = db.Column(db.Text)
    model_id = db.Column(db.Integer)
    inquiry_info = db.Column(db.Integer)


# 药物（药物ID，药品名称，剂量，剂量单位，备注，禁忌，所属处方）
# 主键：药物ID
class DrugInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(10))
    dosage = db.Column(db.Integer)
    unit = db.Column(db.String(10))
    note = db.Column(db.Text)
    taboo = db.Column(db.Text)
    prescription_id = db.Column(db.Integer, ForeignKey('prescription.id'))


# 药物信息sql查询结果转python内置类型函数
def druginfo_to_dict(druginfo):
    dict1 = {
        'name': druginfo.name,
        'dosage': druginfo.dosage,
        'unit': druginfo.unit,
        'note': druginfo.note,
        'taboo': druginfo.taboo,
        'id': druginfo.id
    }
    return dict1


# 数据集（编号，名称，备注，发布时间，发布者，存储路径，查看权限，下载权限，所在文件夹名）
# 主键：编号
class DataSet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128))
    file_name = db.Column(db.String(128))
    note = db.Column(db.Text)
    time = db.Column(db.DateTime)
    path = db.Column(db.Text)
    check_permission = db.Column(db.String(5))
    download_permission = db.Column(db.String(5))
    publisher_name = db.Column(db.String(20))
    folder_name = db.Column(db.Text)
    publisher = db.Column(db.Integer, ForeignKey('user.id'))


# 算法模型（编号，名称，备注，开始训练时间，结束训练时间，数据集，训练算法，存储路径，查看权限，下载权限，发布者）
# 主键：编号
class Model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128))
    note = db.Column(db.Text)
    start_time = db.Column(db.DateTime)
    end_time = db.Column(db.DateTime)
    algorithm = db.Column(db.Text)
    path = db.Column(db.Text)
    check_permission = db.Column(db.String(5))
    download_permission = db.Column(db.String(5))
    publisher = db.Column(db.Integer, ForeignKey('user.id'))
    dataset = db.Column(db.Integer, ForeignKey('data_set.id'))


# 经典方（处方编号，名称，组成药物）
# 主键：经典方编号
class ClassicPrescription(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128))
    drug = db.Column(db.Text)
    similar_degree = db.Column(db.Float)
    similar_prescription = db.Column(db.Integer, ForeignKey('prescription.id'))


# 经典方药物信息（药物ID，药品名称，剂量，剂量单位，备注，禁忌，所属经典方）
# 主键： 药物ID
class ClassicInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(10))
    dosage = db.Column(db.Integer)
    unit = db.Column(db.String(10))
    note = db.Column(db.Text)
    taboo = db.Column(db.Text)
    prescription_id = db.Column(db.Integer, ForeignKey('classic_prescription.id'))


# 当前候选方药物（药物ID，药品名称，剂量，剂量单位，备注，禁忌，所属处方）
# 主键：药物ID
class CandidateDrug(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(10))
    dosage = db.Column(db.Integer)
    unit = db.Column(db.String(10))
    note = db.Column(db.Text)
    taboo = db.Column(db.Text)
    candidate_id = db.Column(db.Integer, ForeignKey('candidate.id'))
    diagnose_id = db.Column(db.Integer, ForeignKey('diagnose.id'))


# 当前候选方（处方编号，名称，组成药物）
# 主键：经典方编号
class Candidate(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    drug = db.Column(db.Text)
    origin_id = db.Column(db.Integer, ForeignKey('prescription.id'))
    diagnose_id = db.Column(db.Integer, ForeignKey('diagnose.id'))


# 诊断书（编号，中医诊断，法则治法，频次，用法备注，给药途径，付数,问诊信息编号，开具处方，医生）
# 主键：编号
class DiagnoseReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tcm_diagnose = db.Column(db.Text)
    fzzf = db.Column(db.Text)
    frequency = db.Column(db.Text)
    note = db.Column(db.Text)
    usage = db.Column(db.Text)
    number = db.Column(db.Integer)
    prescription = db.Column(db.Integer, ForeignKey('prescription.id'))
    inquiry_info = db.Column(db.Integer, ForeignKey('inquiry_info.id'))
    doctor = db.Column(db.Integer, ForeignKey('user.id'))


# 诊断记录（记录号，是否添加人群划分，医生，问诊信息编号，处方编号，诊断书编号，时间）
# 主键：记录号，医生，问诊信息编号，处方编号，诊断书编号
class diagnose(db.Model):
    id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    add_division = db.Column(db.Integer)
    doctor = db.Column(db.Integer, ForeignKey('user.id'))
    inquiryinfo_id = db.Column(db.Integer, ForeignKey('inquiry_info.id'))
    prescription_id = db.Column(db.Integer, ForeignKey('prescription.id'))
    report_id = db.Column(db.Integer, ForeignKey('diagnose_report.id'))
    time = db.Column(db.DateTime)


# 诊断记录sql查询结果转python内置类型函数
def diagnose_to_dict(zd):
    """

    :param zd:
    :return:
    """
    dict1 = {
        'doctor': zd.doctor,
        'inquiryinfo_id': zd.inquiryinfo_id,
        'time': zd.time,
        'id': zd.id
    }
    return dict1


# 人群划分（划分ID，数据名称，划分数 ，备注，发布者，发布时间，状态，预后最好人群）
# 主键：划分ID
class Cluster(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    data_name = db.Column(db.Text)
    cluster_num = db.Column(db.Integer)
    note = db.Column(db.Text)
    time = db.Column(db.DateTime)
    statue = db.Column(db.String(10))
    publisher_name = db.Column(db.String(20))
    best_prognosis = db.Column(db.Integer)
    publisher = db.Column(db.Integer, ForeignKey('user.id'))


# 人群划分信息（人群划分ID，人群号，总人数，平均年龄，性别比例，平均死亡率，症状特征，用药特征）
# 主键：ID
class ClusterInfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    lei = db.Column(db.Integer)
    number = db.Column(db.Integer)
    ave_age = db.Column(db.Float)
    sex_rate = db.Column(db.Text)
    death_rate = db.Column(db.Float)
    symptom = db.Column(db.Text)
    prescription = db.Column(db.Text)
    cluster = db.Column(db.Integer, ForeignKey('cluster.id'))


# 留言板（留言ID，发布者，时间）
# 主键：留言ID
class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    publisher = db.Column(db.Integer, ForeignKey('user.id'))
    time = db.Column(db.DateTime)


# 留言板查询结果转python内置类型函数
def message_to_dict(ly):
    """

    :param zd:
    :return:
    """
    dict1 = {
        'message_id': ly.id,
        'publisher': ly.publisher,
        'time': ly.time
    }
    return dict1


# 留言内容按行分隔（内容ID，内容，所属留言ID）
# 主键：内容ID
class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    word = db.Column(db.Text)
    message_id = db.Column(db.Integer, ForeignKey('message.id'))


# 留言板内容查询结果转python列表
def content_to_list(nr):
    """

    :param zd:
    :return:
    """
    word_list = [nr.word]
    return word_list


# 用户（用户名，密码，姓名，身份）
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    # 资料
    name = db.Column(db.String(20))
    username = db.Column(db.String(20))  # 用户名
    password_hash = db.Column(db.String(128))  # 密码散列值
    real_name = db.Column(db.String(20))  # 真实姓名
    phone = db.Column(db.String(50))  # 联系电话
    email = db.Column(db.String(128))  # 邮箱

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)  # 将生成的密码保持到对应字段

    def validate_password(self, password):  # 用于验证密码的方法，接受密码作为参数
        return check_password_hash(self.password_hash, password)  # 返回布尔值


db.create_all()

login_manager = LoginManager(app)  # 实例化扩展类
login_manager.login_view = 'login'  # 未登录用户访问登录保护视图，跳转到login
login_manager.login_message = u'请登录以访问此界面'


@login_manager.user_loader
def load_user(user_id):
    user = User.query.get(int(user_id))  # 用 ID 作为 User 模型的主键查询对应的用户
    return user


# 进入欢迎页
@app.route('/', methods=['GET', 'POST'])
def welcome():
    if request.method == 'POST':
        return render_template('cftj.html')
    return render_template('index.html')


# 注册
@app.route('/register', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        paw = request.form['paw']
        nc = request.form['nc']
        again = request.form['again']
        if not name or not paw:
            flash('输入错误！')
            return redirect(url_for('signup'))
        # 查询用户名是否已注册
        is_register = User.query.filter_by(username=name).first()
        if is_register:
            flash('该用户名已注册！')
            return redirect(url_for('signup'))
        if again == paw:
            user = User(username=name, name=nc)
            user.set_password(paw)  # 设置密码
            db.session.add(user)
            db.session.commit()
            basepath = os.path.dirname(__file__)  # 当前文件所在路径
            makedir_path = os.path.join(basepath, r'static\uploads\datasets', name)
            os.makedirs(makedir_path)
            flash('注册成功！')
            return redirect(url_for('login'))
        else:
            flash('两次密码不一致！')
            return redirect(url_for('signup'))
    return render_template('signup.html')


# 登录
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if not username or not password:
            flash('输入错误！')
            return redirect(url_for('login'))
        user = User.query.filter_by(username=username).first()
        # 未注册用户提示未注册
        if user:
            if username == user.username and user.validate_password(password):
                login_user(user)  # 登入用户
                return redirect(url_for('home'))  # 重定向到主页
        else:
            flash('该用户名未注册！')
            return redirect(url_for('login'))
        flash('用户名或密码错误！')  # 如果验证失败，显示错误消息
        return redirect(url_for('login'))  # 重定向回登录页面
    return render_template('login.html')


# 退出登录
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('退出成功！')
    return redirect(url_for('welcome'))  # 重定向回欢迎页面


# 进入主页
@app.route('/homepage', methods=['GET', 'POST'])
def home():
    # 读取留言板相关信息
    all_message = Message.query.all()
    message_count = Message.query.count()
    message_list = []
    for m in all_message:
        publisher = User.query.get(m.publisher)
        publisher_name = publisher.username
        mes_list = [publisher_name, m.time]
        # 查询内容
        content_list = []
        content = Content.query.filter_by(message_id=m.id).all()
        for c in content:
            word = content_to_list(c)
            content_list.append(word)
        message_list.append(mes_list + content_list)
    return render_template('home.html', message=message_list, num=message_count)


# 中医处方推荐
@app.route('/cftj', methods=['GET', 'POST'])
@login_required
def recommend():
    if request.method == 'POST':
        # 目前患者ID，由系统默认给出
        # hz_id = request.form.get('id')
        nm = request.form.get('nm')  # 患者姓名
        xb = request.form.get('xb')  # 患者姓名
        nl = request.form.get('age')  # 患者年龄
        blood = request.form.get('blood')  # 患者血型
        hz_id = int(time.time())  # 将当前时间戳作为患者ID

        # 查询数据库，判断患者信息是否已录入，由于现在患者ID是时间戳给出的，不可能出现已录入的情况，此功能目前没有作用
        pat = Patient.query.get(hz_id)
        if pat:  # 若患者信息已录入数据库
            # 更新患者信息
            if nl == '':
                pat.name = nm
                pat.sex = xb
                pat.blood = blood
                db.session.commit()
            else:
                pat.name = nm
                pat.sex = xb
                pat.age = nl
                pat.blood = blood
                db.session.commit()
        else:
            # 将患者信息录入数据库
            if nl == '':
                pat = Patient(id=hz_id, name=nm, sex=xb, blood=blood)
            else:
                pat = Patient(id=hz_id, name=nm, sex=xb, age=nl, blood=blood)
            db.session.add(pat)
            db.session.commit()

        # 测试使用
        # pat = Patient(id=hz_id, name='张三', sex='男', age=50, blood='A')

        # 从前端获取刻下症
        spt = request.form.get('zs')  # 刻下症，一开始使用的是主诉，所以写成zs，后面改为刻下症后，没有修改
        spts = spt.split(';')
        spt_num = len(spts)

        # 从前端获取既往史情况
        family_history = request.form.get('FamilyHistory')
        allergy_history = request.form.get('AllergyHistory')
        alcohol_history = request.form.get('AlcoholHistory')
        smoking_history = request.form.get('SmokingHistory')
        history_list = []

        if family_history == '有':
            history_list.append('家族史')
        if allergy_history == "有":
            history_list.append('过敏史')
        if alcohol_history == '有':
            history_list.append('饮酒史')
        if smoking_history == '有':
            history_list.append('吸烟史')

        past_history = {
            '家族史': family_history,
            '过敏史': allergy_history,
            '饮酒史': alcohol_history,
            '吸烟史': smoking_history
        }

        # 问诊信息传入数据库
        # zz是症状的意思
        zz = InquiryInfo(chief_complaint=spt, patient=hz_id, doctor=current_user.get_id(),
                         past_history=','.join(history_list))
        db.session.add(zz)
        db.session.commit()

        # 当前患者的问诊信息info
        info = InquiryInfo.query.filter_by(patient=hz_id).first()

        # 问诊记录传入数据库
        # wzjl表示问诊记录
        wzjl = Inquiry(patient=hz_id, inquiry_info=info.id, time=datetime.datetime.now(), doctor=current_user.get_id())
        db.session.add(wzjl)
        db.session.commit()

        # 获取前端选择的算法
        mx = request.form.get('sf')

        # 将计算得出的推荐方存入表prescription
        recommend_degree = 0.7  # 默认推荐度为0.7
        pre_name = ''  # 推荐方名称
        similar_symptom = ''  # 相似症状
        is_sim = 0  # 0表示所选算法不是相似度算法，1表示所选算法为相似度算法，由于相似度算法的推荐结果前端界面与其他算法不同，前端根据此变量，生成相应界面

        # 根据不同算法，得出推荐方，并上传数据库
        if mx == 'knn':
            drug_list = onenn(spts)
            drug = Prescription(drug=','.join(drug_list), inquiry_info=info.id)
        elif mx == 'mlknn':
            drug_list = mlknn(spts)
            drug = Prescription(drug=','.join(drug_list), inquiry_info=info.id)
        elif mx == 'tcmpr':
            tcmpr = TcmprModelTest()
            drug_list = tcmpr.main(spts)
            drug = Prescription(drug=','.join(drug_list), inquiry_info=info.id)
        elif mx == 'sim':
            is_sim = 1
            Sim = PrSystemSim()
            top = Sim.main(spt)
            drug_list = top[2].split('、')
            recommend_degree = round(top[3], 2)
            pre_name = top[1]
            similar_symptom = top[0]
            drug = Prescription(drug=','.join(drug_list), inquiry_info=info.id)
        db.session.add(drug)
        db.session.commit()

        # 获取当前患者推荐方编号
        current_prescription = Prescription.query.filter_by(inquiry_info=info.id).first()

        # 将推荐处方中药物信息存入表drug_info，剂量、备注、禁忌都默认给出
        for med in drug_list:
            info = DrugInfo(name=med, dosage=20, unit='mg', note='无', taboo='无',
                            prescription_id=current_prescription.id)
            db.session.add(info)
            db.session.commit()

        # 计算相似经典方
        drug_set = set(drug_list)
        similarity_info = similar(drug_set)
        classic_pre_name = similarity_info[0][0]  # 相似经典方名称
        classic_pre_drugs = similarity_info[1]  # 相似经典方药物组成
        similarity_degree = similarity_info[0][1]  # 相似度

        # 将经典方存入数据库中
        jdf = ClassicPrescription(name=classic_pre_name, drug=','.join(classic_pre_drugs),
                                  similar_degree=similarity_degree, similar_prescription=current_prescription.id)
        db.session.add(jdf)
        db.session.commit()

        # 获取当前经典方
        current_classic_pre = ClassicPrescription.query.filter_by(similar_prescription=current_prescription.id).first()

        # 将经典方药物信息存入数据库中
        for dg in classic_pre_drugs:
            jdf_druginfo = ClassicInfo(name=dg, dosage=20, unit='mg', note='无', taboo='无',
                                       prescription_id=current_classic_pre.id)
            db.session.add(jdf_druginfo)
            db.session.commit()

        # 是否结合生存预测
        d = request.form.get('division')
        cluster_id = request.form.get('cluster')  # 获取选择的人群划分数据ID

        # adddivision，0表示不结合生存预测，1表示结合生存预测，在渲染前端界面时，据此变量，传递不同数据
        if d == '1':
            adddivision = 1
        else:
            adddivision = 0

        # 生成诊断记录
        inquinfo = InquiryInfo.query.filter_by(patient=hz_id).first()
        diagnose_id = int(time.time())
        diag = diagnose(id=diagnose_id, add_division=adddivision, doctor=current_user.id, inquiryinfo_id=inquinfo.id,
                        time=datetime.datetime.now())
        db.session.add(diag)
        db.session.commit()

        # 生成候选方数据表
        candidate_id = int(time.time()) + 66
        candidate = Candidate(id=candidate_id, diagnose_id=diagnose_id)
        db.session.add(candidate)
        db.session.commit()

        if adddivision == 0:
            return render_template('output.html',
                                   Info=DrugInfo.query.filter_by(prescription_id=current_prescription.id).all(),
                                   Pre=current_prescription,
                                   Pat=Patient.query.filter_by(id=hz_id).first(),
                                   Zz=spt, classic=current_classic_pre,
                                   classicinfo=ClassicInfo.query.filter_by(
                                       prescription_id=current_classic_pre.id).all(),
                                   diagnose_id=diagnose_id,
                                   candidate_id=candidate_id, add_division=adddivision, history=past_history,
                                   tjd=recommend_degree, is_sim=is_sim, pre_name=pre_name,
                                   similar_symptom=similar_symptom)
        else:
            # 读取人群信息
            cluster_info = ClusterInfo.query.filter_by(cluster=cluster_id).all()
            spt_list = []
            for i in cluster_info:
                spt_list.append(i.symptom)
            # 计算相似人群
            people_id = compute_label(spt, spt_list) + 1
            # 上传患者所属人群号
            inquinfo.people_id = people_id
            inquinfo.cluster_id = cluster_id
            db.session.commit()

            # 获取相似人群信息
            similar_people = ClusterInfo.query.filter_by(lei=people_id, cluster=cluster_id).first()
            similar_people_num = similar_people.number
            similar_people_age = similar_people.ave_age
            similar_people_sex_rate = similar_people.sex_rate
            similar_people_death_rate = similar_people.death_rate
            similar_people_symptom = similar_people.symptom
            similar_people_prescription = similar_people.prescription

            # 相似人群字典
            people_dict = {
                'people_id': people_id,
                'similar_people_num': similar_people_num,
                'similar_people_age': similar_people_age,
                'similar_people_sex_rate': similar_people_sex_rate,
                'similar_people_death_rate': similar_people_death_rate,
                'similar_people_symptom': similar_people_symptom,
                'similar_people_prescription': similar_people_prescription
            }

            # 获取预后最好人群信息
            current_cluster = Cluster.query.get(cluster_id)
            publisher = current_cluster.publisher_name
            best_prognosis = current_cluster.best_prognosis

            best_people = ClusterInfo.query.filter_by(lei=best_prognosis, cluster=cluster_id).first()
            best_people_num = best_people.number
            best_people_age = best_people.ave_age
            best_people_sex_rate = best_people.sex_rate
            best_people_death_rate = best_people.death_rate
            best_people_symptom = best_people.symptom
            best_people_prescription = best_people.prescription

            # 预后最好人群字典
            best_dict = {
                'lei': best_prognosis,
                'best_people_num': best_people_num,
                'best_people_age': best_people_age,
                'best_people_sex_rate': best_people_sex_rate,
                'best_people_death_rate': best_people_death_rate,
                'best_people_symptom': best_people_symptom,
                'best_people_prescription': best_people_prescription
            }

            return render_template('output.html',
                                   Info=DrugInfo.query.filter_by(prescription_id=current_prescription.id).all(),
                                   Pre=current_prescription,
                                   Pat=Patient.query.filter_by(id=hz_id).first(),
                                   Zz=spt, classic=current_classic_pre,
                                   classicinfo=ClassicInfo.query.filter_by(
                                       prescription_id=current_classic_pre.id).all(),
                                   diagnose_id=diagnose_id,
                                   candidate_id=candidate_id, people_dict=people_dict, add_division=adddivision,
                                   best_people=best_dict, cluster_id=cluster_id, publisher=publisher,
                                   history=past_history, tjd=recommend_degree, is_sim=is_sim, pre_name=pre_name,
                                   similar_symptom=similar_symptom)
    all_cluster = Cluster.query.all()
    return render_template('input.html', cluster=all_cluster)


# 联想搜索
@app.route('/search', methods=['POST'])
def search():
    jg = []
    a = request.get_data()
    s1 = str(a, encoding='utf-8')
    user_dict1 = json.loads(s1)
    for i in symptoms:
        if user_dict1['input'] in i:
            jg.append(i)

    print(jg)
    return jsonify(jg)


# 选择推荐方
# Ajax交互
@app.route('/select/candidate', methods=['POST'])
@login_required
def select():
    a = request.get_data()
    s1 = str(a, encoding='utf-8')  # 解码为string
    pre_dict = json.loads(s1)  # 转化为字典
    pre_id = pre_dict['preid']  # 处方号
    diagnose_id = pre_dict['diagnose_id']  # 诊断记录ID
    candidate_id = pre_dict['candidate_id']  # 候选方ID

    # 选择处方详细信息
    druginfo = DrugInfo.query.filter_by(prescription_id=pre_id).all()

    # 所选处方上传到数据库中
    current_pre = Prescription.query.filter_by(id=pre_id).first()
    current_candidate = Candidate.query.filter_by(id=candidate_id).first()
    current_candidate.drug = current_pre.drug
    current_candidate.origin_id = pre_id
    current_candidate.diagnose_id = diagnose_id
    db.session.commit()

    # 查询是否已经有候选方信息记录
    candidate_info = CandidateDrug.query.filter_by(diagnose_id=diagnose_id).all()
    if candidate_info:
        CandidateDrug.query.filter_by(diagnose_id=diagnose_id).delete()

    # 所选处方详细信息上传到数据库中
    for drug in druginfo:
        info = CandidateDrug(name=drug.name, dosage=drug.dosage, unit=drug.unit, note=drug.note,
                             taboo=drug.taboo, candidate_id=candidate_id, diagnose_id=diagnose_id)
        db.session.add(info)
        db.session.commit()

    candidate = CandidateDrug.query.filter_by(diagnose_id=diagnose_id).all()

    druginfo_list = []
    for item in candidate:
        druginfo_dict = druginfo_to_dict(item)
        druginfo_list.append(druginfo_dict)
    return json.dumps(druginfo_list)


# 选择经典方
# Ajax交互
@app.route('/select/classic', methods=['POST'])
@login_required
def classic():
    a = request.get_data()
    s1 = str(a, encoding='utf-8')  # 解码为string
    pre_dict = json.loads(s1)  # 转化为字典
    pre_id = pre_dict['preid']  # 处方号
    diagnose_id = pre_dict['diagnose_id']  # 诊断记录ID
    candidate_id = pre_dict['candidate_id']  # 候选方ID

    # 选择处方详细信息
    druginfo = ClassicInfo.query.filter_by(prescription_id=pre_id).all()

    # 所选处方上传到数据库中
    current_pre = ClassicPrescription.query.filter_by(similar_prescription=pre_id).first()
    current_candidate = Candidate.query.filter_by(id=candidate_id).first()
    current_candidate.drug = current_pre.drug
    current_candidate.origin_id = pre_id
    current_candidate.diagnose_id = diagnose_id
    db.session.commit()

    # 查询是否已经有候选方信息记录
    candidate_info = CandidateDrug.query.filter_by(diagnose_id=diagnose_id).all()
    if candidate_info:
        CandidateDrug.query.filter_by(diagnose_id=diagnose_id).delete()

    # 所选处方详细信息上传到数据库中
    for drug in druginfo:
        info = CandidateDrug(name=drug.name, dosage=drug.dosage, unit=drug.unit, note=drug.note,
                             taboo=drug.taboo, candidate_id=candidate_id, diagnose_id=diagnose_id)
        db.session.add(info)
        db.session.commit()

    candidate = CandidateDrug.query.filter_by(diagnose_id=diagnose_id).all()
    druginfo_list = []
    for item in candidate:
        druginfo_dict = druginfo_to_dict(item)
        druginfo_list.append(druginfo_dict)
    return json.dumps(druginfo_list)


# 清空候选方
# Ajax交互
@app.route('/candidate/clear', methods=['POST'])
@login_required
def clear():
    a = request.get_data()
    s1 = str(a, encoding='utf-8')  # 解码为string
    pre_dict = json.loads(s1)  # 转化为字典
    diagnose_id = pre_dict['diagnose_id']
    print(diagnose_id)
    # 删除候选方记录药品
    candidate = Candidate.query.filter_by(diagnose_id=diagnose_id).first()
    candidate.drug = ''
    # 删除候选方信息记录
    CandidateDrug.query.filter_by(diagnose_id=diagnose_id).delete()
    db.session.commit()
    return json.dumps({'a': 1})


# 添加候选方药品
# Ajax交互
@app.route('/candidate/add', methods=['POST'])
@login_required
def add():
    a = request.get_data()
    s1 = str(a, encoding='utf-8')  # 解码为string
    add = json.loads(s1)  # 转化为字典
    print(add)
    print(type(int(add['diagnose_id'])))
    # 向候选方数据表记录中添加药品
    candidate = Candidate.query.filter_by(diagnose_id=int(add['diagnose_id'])).first()
    # 根据表中是否为空判断是否需要加逗号
    if candidate.drug:
        candidate.drug += ',' + add['name']
    else:
        candidate.drug = str(add['name'])
    db.session.commit()

    # 向候选方药品信息表中添加记录
    hxinfo = CandidateDrug(name=add['name'], dosage=add['dosage'], unit=add['unit'], note=add['note'],
                           taboo=add['taboo'], candidate_id=add['candidate_id'], diagnose_id=add['diagnose_id'])
    db.session.add(hxinfo)
    db.session.commit()
    candi = CandidateDrug.query.filter_by(diagnose_id=int(add['diagnose_id'])).all()
    last = candi[-1]
    return json.dumps({'drugid': last.id})


# 删除候选方药品
# Ajax交互
@app.route('/candidate/delete', methods=['POST'])
@login_required
def deletedrug():
    a = request.get_data()
    s1 = str(a, encoding='utf-8')  # 解码为string
    data_dict = json.loads(s1)  # 转化为字典
    diagnose_id = data_dict['diagnose_id']
    drug_id = data_dict['drug_id']
    # 获取药物名称
    drug = CandidateDrug.query.get(drug_id)
    drugname = drug.name

    # 删除候选方记录药品
    candidate = Candidate.query.filter_by(diagnose_id=diagnose_id).first()
    drugstr = str(candidate.drug)
    drugstr = drugstr.replace(',' + drugname, '')
    drugstr = drugstr.replace(drugname + ',', '')
    drugstr = drugstr.replace(str(drugname), '')
    candidate.drug = drugstr
    # 删除候选方信息记录
    CandidateDrug.query.filter_by(id=drug_id).delete()
    db.session.commit()
    return json.dumps({'a': 1})


# 诊断
@app.route('/diagnose/<int:zdid>', methods=['GET', 'POST'])
@login_required
def zd(zdid):
    if request.method == 'POST':
        # 获取前端表单数据
        use_way = request.form.get('use_way')  # 给药途径
        use_note = request.form.get('use_note')  # 用法备注
        use_time = request.form.get('use_time')  # 频次
        num = request.form.get('num')  # 付数

        # 查询当前诊断记录
        current_diagnose = diagnose.query.get(zdid)

        # 获取医生信息
        doctor_id = current_diagnose.doctor  # 诊断医生ID
        doctor = User.query.get(doctor_id)  # 查询当前医生用户
        doctor_name = doctor.username  # 医生姓名

        # 获取患者信息
        inquiryinfo_id = current_diagnose.inquiryinfo_id  # 当前问诊信息ID
        inquiryinfo = InquiryInfo.query.get(inquiryinfo_id)  # 查询当前问诊信息
        patient_id = inquiryinfo.patient  # 当前患者ID
        patient = Patient.query.get(patient_id)  # 当前患者
        patient_name = patient.name  # 患者姓名
        patient_sex = patient.sex  # 患者性别
        patient_age = patient.age  # 患者年龄

        # 将诊断书相关数据提交到数据库
        report = DiagnoseReport(frequency=use_time, note=use_note, usage=use_way, number=num,
                                inquiry_info=inquiryinfo_id, doctor=doctor_id)
        db.session.add(report)
        db.session.commit()
        return redirect(url_for('zd', zdid=zdid))

    # 查询当前诊断记录
    current_zd = diagnose.query.get(zdid)
    zd_time = current_zd.time  # 诊断时间
    # 获取医生信息
    doctor_id = current_zd.doctor  # 诊断医生ID
    doctor = User.query.get(doctor_id)  # 查询当前医生用户
    doctor_name = doctor.username  # 医生姓名
    # 获取患者信息
    inquiryinfo_id = current_zd.inquiryinfo_id  # 当前问诊信息ID
    inquiryinfo = InquiryInfo.query.get(inquiryinfo_id)  # 查询当前问诊信息
    patient_id = inquiryinfo.patient  # 当前患者ID
    patient = Patient.query.get(patient_id)  # 当前患者
    patient_name = patient.name  # 患者姓名
    patient_sex = patient.sex  # 患者性别
    patient_age = patient.age  # 患者年龄
    # 获取候选方信息
    candidate_drug = CandidateDrug.query.filter_by(diagnose_id=zdid).all()

    # 获取诊断书信息
    report = DiagnoseReport.query.filter_by(inquiry_info=inquiryinfo_id).first()
    use_way = report.usage  # 用法途径
    use_time = report.frequency  # 频次
    use_note = report.note  # 用法备注
    num = report.number  # 付数

    return render_template('report.html', patient_name=patient_name, patient_sex=patient_sex, patient_age=patient_age,
                           patient_id=patient_id, doctor_name=doctor_name, zd_time=zd_time, drugs=candidate_drug,
                           use_way=use_way, use_time=use_time, use_note=use_note, number=num)


# 数据集上传
@app.route('/dataset', methods=['GET', 'POST'])
@login_required
def dataset():
    if request.method == 'POST':
        # 获取前端表单数据
        f = request.files.get('inputfile')  # 数据集文件
        dataname = request.form.get('name')  # 文件名
        remark = request.form.get('note')  # 备注
        access = request.form.get('access')  # 查看权限
        action_access = request.form.get('action_access')  # 下载权限

        # 存储上传的数据集文件
        user = current_user.username  # 当前用户
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        time = datetime.datetime.now()
        format_time = str(time).replace(':', '.')
        mkdir_path = os.path.join(basepath, r'static\uploads\datasets', user, format_time)
        os.makedirs(mkdir_path)
        upload_path = os.path.join(mkdir_path, f.filename)
        f.save(upload_path)  # 保存路径为.\static\uploads\username\time

        # 上传数据集文件信息传入数据库
        fl = DataSet(name=dataname, note=remark, time=time, path=upload_path,
                     check_permission=access, download_permission=action_access, publisher=current_user.id,
                     publisher_name=current_user.username, file_name=f.filename, folder_name=format_time)
        db.session.add(fl)
        db.session.commit()
        return redirect(url_for('dataset'))
    file_list = DataSet.query.all()
    return render_template('dataset.html', file_list=file_list)


# 数据集删除
@app.route('/dataset/delete/<format_time>/<int:file_id>', methods=['GET', 'POST'])
@login_required
def delete(format_time, file_id):
    fd = DataSet.query.get_or_404(file_id)
    basepath = os.path.dirname(__file__)
    filepath = os.path.join(basepath, r'static\uploads\datasets', fd.publisher_name, format_time, fd.file_name)
    os.remove(filepath)
    dir_path = os.path.join(basepath, r'static\uploads\datasets', fd.publisher_name, format_time)
    os.rmdir(dir_path)
    db.session.delete(fd)
    db.session.commit()
    return redirect(url_for('dataset'))

# 进入人群划分界面
@app.route('/divide', methods=['GET', 'POST'])
def divide():
    all_cluster = Cluster.query.all()

    return render_template('people_divide.html', all_cluster=all_cluster)


# 人群划分
# Ajax交互
@app.route('/cluster', methods=['GET', 'POST'])
def cluster():
    # 获取前端数据
    min_lei = 0
    f = request.files.get('inputfile')  # 患者数据文件
    dataname = request.form.get('dataname')  # 数据名称
    remark = request.form.get('note')  # 备注
    num = request.form.get('num')  # 划分数

    # 人群划分ID
    cluster_id = int(time.time())

    # 创建文件夹
    folder_name = str(cluster_id) + current_user.username
    basepath = os.path.dirname(__file__)  # 当前文件所在路径
    makedir_path = os.path.join(basepath, r'static\cluster', folder_name)
    os.makedirs(makedir_path)
    data_path = os.path.join(basepath, r'static\cluster', folder_name, 'data')
    os.makedirs(data_path)  # 初始数据文件
    result_path = os.path.join(basepath, r'static\cluster', folder_name, 'result')
    os.makedirs(result_path)  # 结果文件
    image_path = os.path.join(basepath, r'static\cluster', folder_name, 'image')
    os.makedirs(image_path)  # 图像文件

    # 保存原始数据
    upload_path = os.path.join(data_path, 'data.xlsx')
    f.save(upload_path)  # 保存路径为.\static\cluster\fold\data

    # 检查文件格式
    # is_correct包含是否有误、错误信息
    is_correct = check(upload_path)

    if is_correct[0] == 1:
        # 将人群划分数据上传到数据库中
        c = Cluster(id=cluster_id, data_name=dataname, cluster_num=num, note=remark, time=datetime.datetime.now(),
                    publisher=current_user.id, statue='NO', publisher_name=current_user.username)
        db.session.add(c)
        db.session.commit()

        # 进行划分和生存预测
        clustering = Clustering2Analysis(upload_path)
        clustering.main(int(num), result_path, image_path)

        # 更改人群划分状态
        current_cluster = Cluster.query.get(cluster_id)
        current_cluster.statue = 'OK'
        db.session.commit()

        # 将人群划分结果上传到数据库中
        final_path = result_path + r'\Result_final.xlsx'
        df = pd.read_excel(final_path)
        temp = 1
        for i in list(df.values):
            lei_id = i[0]
            number = i[3]
            sex_rate = '男' + str(i[4]) + '/' + '女' + str(i[5])
            ave_age = i[6]
            death_rate = i[12]
            if death_rate < temp:
                temp = death_rate
                min_lei = lei_id
            symptom = i[1]
            prescription = i[2]
            info = ClusterInfo(lei=lei_id + 1, number=number, ave_age=ave_age, sex_rate=sex_rate, death_rate=death_rate,
                               symptom=symptom, prescription=prescription, cluster=cluster_id)
            db.session.add(info)
            db.session.commit()
        current_cluster.best_prognosis = min_lei + 1
        db.session.commit()

        jg = {
            'status': 1,
            'message': ''
        }
        return json.dumps(jg)
    else:
        # 删除初始数据
        os.remove(upload_path)

        # 删除各项文件夹
        os.rmdir(data_path)
        os.rmdir(result_path)
        os.rmdir(image_path)
        os.rmdir(makedir_path)
        jg = {
            'status': 0,
            'message': is_correct[1]
        }
        return json.dumps(jg)


# 人群划分界面定时刷新
# Ajax交互
# 使得划分完成时，前端刷新为已完成
@app.route('/cluster/refresh', methods=['GET', 'POST'])
def refresh():
    finished = Cluster.query.filter_by(statue='OK').all()
    finished_id_list = []
    for i in finished:
        finished_id_list.append(i.id)
    return json.dumps(finished_id_list)


# 人群划分删除
@app.route('/cluster/delete/<int:cluster_id>/<username>')
def cluster_delete(cluster_id, username):
    basepath = os.path.dirname(__file__)  # 当前文件所在路径
    folder_name = str(cluster_id) + username
    dir_path = os.path.join(basepath, r'static\cluster', folder_name)

    # 删除result文件夹及内部文件
    result_path = os.path.join(dir_path, 'result')
    for i in os.listdir(result_path):
        result_file_path = os.path.join(result_path, i)
        os.remove(result_file_path)
    os.rmdir(result_path)

    # 删除data文件夹及内部文件
    data_path = os.path.join(dir_path, 'data')
    for i in os.listdir(data_path):
        data_file_path = os.path.join(data_path, i)
        os.remove(data_file_path)
    os.rmdir(data_path)

    # 删除image文件夹及内部文件
    image_path = os.path.join(dir_path, 'image')
    for i in os.listdir(image_path):
        image_file_path = os.path.join(image_path, i)
        os.remove(image_file_path)
    os.rmdir(image_path)

    # 删除此划分文件夹
    os.rmdir(dir_path)

    # 删除数据库中划分详细信息数据
    ClusterInfo.query.filter_by(cluster=cluster_id).delete()
    db.session.commit()

    # 删除划分记录
    c = Cluster.query.get(cluster_id)
    db.session.delete(c)
    db.session.commit()

    return redirect(url_for('divide'))


# 人群划分详细结果
@app.route('/<cluster_id>/info', methods=['GET', 'POST'])
def people(cluster_id):
    info = ClusterInfo.query.filter_by(cluster=cluster_id).all()
    c = Cluster.query.get(cluster_id)
    return render_template('people_divide_info.html', info=info, cluster=c)


# 模型管理
@app.route('/model', methods=['GET', 'POST'])
@login_required
def model():
    return render_template('model.html')


# 诊断日志
@app.route('/diagnose/log', methods=['GET', 'POST'])
@login_required
def log():
    # 诊断日志列表
    log_list = []
    # 获取所有诊断记录
    all_diagnose = diagnose.query.all()
    for i in all_diagnose:
        # 转化为字典格式
        item = diagnose_to_dict(i)
        # 问诊信息编号
        inquiry_id = item['inquiryinfo_id']
        # 查询患者信息
        info = InquiryInfo.query.get(inquiry_id)
        patient_id = info.patient
        patient = Patient.query.get(patient_id)
        patient_name = patient.name
        patient_sex = patient.sex
        # 就诊时间
        diagnose_time = item['time']

        # 就诊医生
        doctor_id = item['doctor']
        doctor = User.query.get(doctor_id)
        doctor_name = doctor.username

        # 判断就诊是否完成
        report = DiagnoseReport.query.filter_by(inquiry_info=inquiry_id).first()
        if report == None:
            statue = 0
        else:
            statue = 1

        # 诊断日志字典
        diagnose_dict = {
            'id': item['id'],
            'name': patient_name,
            'sex': patient_sex,
            'time': diagnose_time,
            'doctor': doctor_name,
            'statue': statue
        }
        # 将字典存入列表中
        log_list.append(diagnose_dict)

    return render_template('log.html', log=log_list)


# 用户主页
@app.route('/user/home', methods=['GET', 'POST'])
@login_required
def center():
    """

    :return:
    """
    return render_template('user_home.html')


# 修改个人信息
@app.route('/user/edit', methods=['GET', 'POST'])
def edit():
    if request.method == 'POST':
        current_user.name = request.form.get('nc')
        current_user.real_name = request.form.get('real_name')
        current_user.phone = request.form.get('phone')
        current_user.email = request.form.get('email')
        db.session.commit()
        flash('修改成功！')
        return redirect('edit')
    return render_template('user_home.html')


# 重置密码
@app.route('/password/change', methods=['GET', 'POST'])
def change():
    if request.method == 'POST':
        old = request.form['old']
        new = request.form.get('new')
        again = request.form.get('new_again')
        if current_user.validate_password(old):
            if new == again:
                current_user.set_password(new)  # 设置新密码
                db.session.commit()
                flash('修改成功！')
                return redirect(url_for('login'))  # 重定向到登录界面
            else:
                flash('两次密码输入不一致！')
                return render_template('user_home.html')

        else:
            flash('旧密码不匹配！')  # 如果验证失败，显示错误消息
            return render_template('user_home.html')  # 重定向回个人主页


# 查看他人主页
@app.route('/<username>/home', methods=['GET', 'POST'])
def otherhome(username):
    user = User.query.filter_by(username=username).first()
    return render_template('other_home.html', user=user)


# 发布留言
@app.route('/message', methods=['GET', 'POST'])
def message():
    if request.method == 'POST':
        ly = request.form.get('ly')
        word_list = ly.split('\n')
        message_id = int(time.time())
        mes = Message(id=message_id, publisher=current_user.id, time=datetime.datetime.now())
        db.session.add(mes)
        db.session.commit()
        for w in word_list:
            content = Content(word=w, message_id=message_id)
            db.session.add(content)
            db.session.commit()
        return redirect('homepage')
    return redirect('homepage')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
