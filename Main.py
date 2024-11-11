import os
import random
import re
import subprocess
import threading
import time
from time import sleep

import telebot
import webbrowser
import pyautogui
import pyaudio
import keyboard
import torch
import openai
import json
import soundfile as sf
import sounddevice as sd
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import QThread,pyqtSignal
from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout, QLabel
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QApplication, QDialog
from PyQt5.QtWidgets import QWidget
from sympy.codegen.cfunctions import expm1
from vosk import Model, KaldiRecognizer
from PyQt5 import QtWidgets, QtCore, uic,QtGui
from PyQt5.QtWidgets import (QFileDialog)

from Sort_and_replace_text import Replacing_The_Text


# Загрузка UI файла
Ui_Form, _ = uic.loadUiType('VoiceConvAsis_U_3I.ui')
Ui_Form_sub, _ = uic.loadUiType('Threaded_sub_window.ui')
temp_vol = -1
Chat_message_TG_bot = ''
file_name = 'OpenAiApiKey.txt'

if not os.path.exists(file_name):
    open(file_name, 'w',encoding='utf-8').close()
with open(file_name, 'r',encoding='utf-8') as file:
    OpenAi_ApiKey = file.read()
openai.api_key = OpenAi_ApiKey


file_name_tg ="Telegram_API.txt"
if not os.path.exists(file_name_tg):
    open(file_name_tg, 'w',encoding='utf-8').close()
with open(file_name_tg, 'r',encoding='utf-8') as file:
    telegram_api = file.read()
TG_API = telegram_api
print(telegram_api)
with open('savefile.json', 'r', encoding='utf-8') as json_file:
    parameter_save_file = json.load(json_file)
Number_of_tokens = parameter_save_file['Number_of_tokens']
Temperature = parameter_save_file['Temperature']

Min_STime_Question = parameter_save_file['Question_Interval_min']
Max_STime_Question = parameter_save_file['Question_Interval_max']

Manner_Voice_CP = parameter_save_file['Manner_Voice_CP']
History_Mem_CP = parameter_save_file['History_Mem_CP']
Personality_CP = parameter_save_file['Personality_CP']

Speaker_Voice = parameter_save_file['Speaker_Voice']
Saved_Sites = parameter_save_file['Saved_Sites']
Saved_Prog = parameter_save_file['Saved_Prog']

Sub_Pos_W= parameter_save_file['Sub_Pos']
Sub_Pos = parameter_save_file['Sub_Pos_XY']
X_axis = parameter_save_file['Xaxis']
Y_axis = parameter_save_file['Yaxis']


class ChatMemory:
    def __init__(self, max_messages=10):
        self.max_messages = max_messages
        self.messages = []

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)

    def get_messages(self):
        return self.messages

    def load_messages(self, chat_history):
        self.messages = chat_history

def generate_response(messages):
    global Number_of_tokens, Temperature
    response = openai.ChatCompletion.create(
        model='gpt-4o-mini',  # Замените на вашу модель
        messages=messages,
        max_tokens=Number_of_tokens,
        temperature=Temperature
    )
    return response.choices[0].message['content'].strip()

def generate_response_history(messages):
    global Number_of_tokens
    response = openai.ChatCompletion.create(
        model='gpt-4o-mini',  # Замените на вашу модель
        messages=messages,
        max_tokens=Number_of_tokens*3,
        temperature=0.5
    )
    return response.choices[0].message['content'].strip()

class Sub_Win(QMainWindow):
    def __init__(self, thread):
        super().__init__()
        try:
            self.ui = uic.loadUi("Threaded_sub_window.ui", self)
            self.thread = thread
            screen = QApplication.primaryScreen()
            screen_size = screen.size()

            self.setWindowFlag(Qt.WindowStaysOnTopHint)
            self.setAttribute(Qt.WA_TranslucentBackground, True)
            self.setWindowFlags(Qt.FramelessWindowHint)
            self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
            self.move(Sub_Pos[0],Sub_Pos[1])
            self.screen_heights = {720:0, 1080:4, 1440:8, 2160:12, 2880:16, 4320:20}

            self.label.setStyleSheet('color:rgb(220, 220, 220);'
                                     'background:rgba(40,40,40,0);'
                                     'padding-left:10px;'
                                     'font-family: "TimesNewRoman";'
                                     f'font-size:{20+self.screen_heights[int(screen_size.height())]}px;')

            self.thread.send_param.connect(self.update_label)
        except Exception as e:
            print(f"Ошибка при инициализации Sub_Win: {e}")

    def update_label(self, text):
        self.label.setText(text)

class ThreadWindow(QThread):
    threadSignal = pyqtSignal(int)
    send_param = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.running = True
        self.message = ''

    def set_message(self, text):
        self.message = text

    def run(self):
        self.send_param.emit(f"Сообщение {self.message}")  # Отправляем текст

    def stop(self):
        self.running = False

class TelegramBot:
    def __init__(self, window):
        global TG_API
        self.window = window
        self.bot = telebot.TeleBot(TG_API)
        self.allowed_users = [945985582]
        self.fifteens_times_history_num = 0

        self.start_bot()

    def start_bot(self):
        thread = threading.Thread(target=self.polling)
        thread.start()

    def polling(self):
        @self.bot.message_handler(commands=["start"])
        def start(m, res=False):
            try:
                if m.from_user.id in self.allowed_users:
                    self.bot.send_message(m.chat.id, 'Я на связи. Напиши мне что-нибудь.')
                else:
                    print(m.from_user.id)
                    self.bot.send_message(m.chat.id, 'Я не на связи. У вас нет доступа.')
            except Exception as f:
                print(f)

        @self.bot.message_handler(func=lambda message: True)
        def handle_message(message):

            global Manner_Voice_CP , History_Mem_CP,Personality_CP,parameter_save_file,Chat_message_TG_bot
            if message.from_user.id not in self.allowed_users:
                self.bot.send_message(message.chat.id, 'У вас нет доступа.')
                return
            ChatHistory = parameter_save_file["Chat_history"]
            if ChatMemory:
                memory = ChatMemory()
                memory.load_messages(ChatHistory)  # Загрузка истории в память
            else:
                memory = ChatMemory()
            set_history_text = False

            if self.fifteens_times_history_num == 10:
                self.fifteens_times_history_num =0
                chat_history = memory.get_messages()
                chat_string = "\n".join(f"{entry['role']}: {entry['content']}" for entry in chat_history)
                text = (
                    'сократи информацию из всего текста (в частности у пользователя) такую что может пригодиться на долгое время, что можно вcпомнить на будущее.'
                    'и то что попросил или потребовал пользователь у ассистента. сократить все очень кратко по возможности в одного предложения. пиши словно идет обращение к ассистенту. при отсутствии важной инфорvации вывести пустоту.'
                    'нужен только текст о том какой ассистент должен быть с тем, что должен помнить и делать')
                sys_message = f"'{chat_string}' - {text}"
                messages_with_system = [{"role": "system", "content": sys_message}]
                response = generate_response_history(messages_with_system)
                print(">>>", response, "<<<")
                messages_with_system = [{"role": "system", "content": (
                    f"'{response}' - добавь информацию из первого текста в этот '{History_Mem_CP}' адаптируя первый текст под стиль написании второго и также изменяй второй текст добавляя информацию из первого, или просто добавь из первого. пиши словно идет обращение к ассистенту. текст должен быть написано к обращению ассистенту."
                    f"должно быть все написано обезличено без упоминания пользвателя. все должно быть без противоречий в готовом тексте, если они есть то удалить или изменить в готовом тексте. выведи только в один текст без вопросительных и восклицательных знаков, соответсвенно без вопросов и восклицания. нужен текст-модель общения")}]
                response = generate_response_history(messages_with_system)
                print("<<<", response, ">>>")
                History_Mem_CP = response
                parameter_save_file['History_Mem_CP'] = History_Mem_CP
                set_history_text =True

                self.window.user_input_signal_history.emit('1111')


            self.fifteens_times_history_num +=1
            print(self.fifteens_times_history_num)
            parameter_save_file["Chat_history"] = memory.get_messages()
            with open('savefile.json', 'w', encoding='utf-8') as json_file:
                json.dump(parameter_save_file, json_file, ensure_ascii=False, indent=len(parameter_save_file))
            print('savefile.json saved in folder for TelegramBot')

            system_message = History_Mem_CP + Personality_CP + Manner_Voice_CP
            print(system_message)
            user_input = message.text
            memory.add_message(role='user', content=user_input)
            messages_with_system = [{"role": "system", "content": system_message}] + memory.get_messages()

            response = generate_response(messages_with_system)
            Chat_message_TG_bot = f"Пользователь: {user_input}\nБот: {response}"
            self.window.user_input_signal.emit(user_input)
            memory.add_message(role='assistant', content=response)


            try:
                print('tg-bot class' + Chat_message_TG_bot )
                self.bot.send_message(message.chat.id, response)
            except Exception as e:
                print(e)
        self.bot.polling(none_stop=True)

    def handle_text_signal(self, text):
            None
class Window(QtWidgets.QMainWindow, Ui_Form):
    text_signal = QtCore.pyqtSignal(str)
    user_input_signal = QtCore.pyqtSignal(str)
    user_input_signal_history = QtCore.pyqtSignal(str)
    temp_vol = -1
    initialized_loadfile = False


    def __init__(self):
        super(Window, self).__init__()
        self.setupUi(self)
        """self.setWindowFlag(Qt.FramelessWindowHint)"""
        self.system_message = Personality_CP + History_Mem_CP + Manner_Voice_CP

        self.dict_word_used_browser=[    "Вы выбрали браузер",
                                         "Вы определились с браузером",
                                         "Вы остановились на браузере",
                                         "Вы выбрали этот браузер",
                                         "Вы решили использовать браузер",
                                         "Вы выбрали данный браузер",
                                         "Вы указали браузер",
                                         "Вы предпочли этот браузер",
                                         "Вы выбрали данный вариант браузера",
                                         "Вы установили браузер"]
        self.browsers_executables = [
            'chrome.exe',        # Google Chrome
            'firefox.exe',       # Mozilla Firefox
            'msedge.exe',        # Microsoft Edge
            'iexplore.exe',      # Internet Explorer
            'opera.exe',         # Opera
            'brave.exe',         # Brave Browser
            'vivaldi.exe',       # Vivaldi
            'browser.exe',       # Yandex Browser
            'maxthon.exe',       # Maxthon
            'qutebrowser.exe',   # QuteBrowser (если используется на Windows)
            'palemoon.exe',      # Pale Moon
            'waterfox.exe',      # Waterfox
            'otter.exe',         # Otter Browser
        ]

        self.Style_Sheet_act = ('QPushButton {'
                           'background: rgb(20, 20, 20);'
                           'color: white;'
                           'border-radius: 15px;'
                           'font-family:System;'
                           'font-size:12px;}'

                           'QPushButton :hover {'
                           'background: rgb(20, 20, 20);'
                           'color: white;'
                           'border-radius: 15px;'
                           'font-family:System;'
                           'font-size:12px;}'

                           'QPushButton :pressed {'
                           'background: rgb(10, 10, 10);}')
        self.Style_Sheet_norm = ('QPushButton {'
                            'background: rgb(40, 40, 40);'
                            'color: white;'
                            'border-radius: 15px;'
                            'font-family:System;'
                            'font-size:12px;}'

                            'QPushButton :hover {'
                            'background: rgb(20, 20, 20);'
                            'color: white;'
                            'border-radius: 15px;'
                            'font-family:System;'
                            'font-size:12px;}'

                            'QPushButton :pressed {'
                            'background: rgb(10, 10, 10);}')

        self.pushButton_2.clicked.connect(self.startVoiceRecognition)
        self.pushButton_2.toggled.connect(self.startVoiceRecognition)
        self.pushButton_2.setIconSize(QtCore.QSize(24, 24))

        self.deactivate_voice.clicked.connect(self.off_or_on_sint_voice)
        self.deactivate_voice.toggled.connect(self.off_or_on_sint_voice)
        self.deactivate_voice.setIconSize(QtCore.QSize(20, 20))

        self.icon_normal = QtGui.QIcon('icon/img_11728.png')
        self.icon_active = QtGui.QIcon('icon/img_11728_white.png')
        self.icon_voice_active = QtGui.QIcon('icon/sound_asis_on.png')
        self.icon_voice_normal = QtGui.QIcon('icon/sound_asis_off.png')

        self.pushButton_2.setIcon(self.icon_normal)
        self.deactivate_voice.setIcon(self.icon_voice_normal)
        self.check_activate_sint_voice = False

        self.pushButton_4.clicked.connect(self.handle_input)
        self.lineEdit_2.returnPressed.connect(self.input_Massage)

        self.SaveApiSet_But.clicked.connect(self.Save_OpenAI_settings)
        self.Question_Interval_saveButton.clicked.connect(self.Save_Ttaaq)

        # func - sites
        self.Save_sites_info_PButton.clicked.connect(lambda :self.Save_Sites_Adress_Name("Site"))
        self.Clear_list_Button.clicked.connect(lambda: self.ClearList_Sites("Site"))
        self.Disagreement_CSitesList_button.clicked.connect(lambda :self.ClearList_SitesProg_side(self.Confirmation_clearing_site_list))
        self.clear_site_list_confirm = 0

        # func - program
        self.Save_prog_info_PButton.clicked.connect(lambda :self.Save_Sites_Adress_Name("Prog"))
        self.PathTo_adress_prog.clicked.connect(self.PathTo_setAdressProg)
        self.Disagreement_CProgList_button.clicked.connect(lambda :self.ClearList_SitesProg_side(self.Confirmation_clearing_prog_list))
        self.Clear_listP_Button.clicked.connect(lambda: self.ClearList_Sites("Prog"))
        self.clear_prog_list_confirm = 0

        self.confirm_clear_SitesLists = 0
        self.confirm_clear_ProgLists = 0


        self.file_path = parameter_save_file['Browser_directory']
        self.save_file = self.file_path
        self.ChatHistory = parameter_save_file["Chat_history"]
        print('> file ', self.save_file, type(self.save_file), "uploaded...")
        print(f"> The chat history has been uploaded. The data file {self.ChatHistory}...")

        if ChatMemory:
            self.memory = ChatMemory()
            self.memory.load_messages(self.ChatHistory)  # Загрузка истории в память
        else:
            self.memory = ChatMemory()

        # значения переменных - values of variables
        self.temp = None
        self.y_t_temp = None
        self.tab_temp = 0
        self.number_range = 0
        self.massage_text =''
        self.default_browser_state = 0
        self.out_text = ""
        self.model=None
        self.history_edit_check = 0



        # озвучка текста -
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_num_threads(8)
        self.local_file = 'model.pt'

        if not hasattr(torch, 'cached_model'):
            if not os.path.isfile(self.local_file):
                print("> Скачивание модели...")
                torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v4_ru.pt', self.local_file)

            print("> Загрузка модели...")
            self.model = torch.package.PackageImporter(self.local_file).load_pickle("tts_models", "model")
            self.model.to(self.device)
            torch.cached_model = self.model
        else:
            self.model = torch.cached_model

        #---------
        self.text_signal.connect(self.updateTextBrowser)

        # кнопки - buttons
        self.button_options.clicked.connect(lambda : self.Slide_Frame_Options())
        self.toolButton_3.clicked.connect(lambda: self.Slide_Frame_Main())
        self.func_edit_Button.clicked.connect(lambda : self.Slide_Frame_Func_edit())
        self.Personality_Sett_TE_Button.clicked.connect(lambda : self.Anim_Slide_Frame_Pers_TE())
        self.History_Sett_TE_Button.clicked.connect(lambda : self.Anim_Slide_Frame_History_TE())
        self.Manner_Sett_TE_Button.clicked.connect(lambda: self.Anim_Slide_Frame_Manner_TE())

        self.baya_tButton.clicked.connect(lambda: self.Change_Voice_Speaker('baya'))
        self.kseniya_tButton.clicked.connect(lambda : self.Change_Voice_Speaker('kseniya'))
        self.xenia_tButton.clicked.connect(lambda:self.Change_Voice_Speaker('xenia'))
        self.aidar_tButton.clicked.connect(lambda:self.Change_Voice_Speaker('aidar'))
        self.eugene_tButton.clicked.connect(lambda:self.Change_Voice_Speaker('eugene'))

        self.Save_Tg_API.clicked.connect(lambda:self.Save_tgAPI())

        self.side_Tg_bot_Button.clicked.connect(self.side_Tg_bot_consol)
        self.TgB_Manner = 0

        # анимация и значения размера контейнеров - animation and container size values
        self.Side_Menu_Num = 0 # окно настроек
        self.Side_Menu_Num_2 = 0 # боковое меню
        self.Side_Menu_Num_FE = 0 # окно функций
        self.Side_Menu_Num_Consol = 1
        self.ASF_Pers_TE = 0
        self.ASF_History_TE = 0
        self.ASF_Manner_TE = 0

        self.frame_6_max_width = 720
        self.frame_6_min_width = 480
        self.frame_6_main_width = 720
        self.animation_block = 0
        self.duration_anim_sideMenu = 200

        self.path_to_browser.clicked.connect(self.open_browser_file)
        self.browser_close_1.clicked.connect(self.delete_save_browser_default)

        if not self.initialized_loadfile:
            self.load_save_file()
            self.initialized_loadfile = True

        # таймер - timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.execute_command)
        self.timer_interval_set()
        self.timer.start(self.interval)

        # позиция субтитров
        self.Sub_Pos_SaveButton.clicked.connect(lambda: self.SubPos_Save())

        self.sub_pos_NW_Button.clicked.connect(lambda: self.Position_Subtitles_Set('nw'))
        self.sub_pos_N_Button.clicked.connect(lambda: self.Position_Subtitles_Set('n'))
        self.sub_pos_NE_Button.clicked.connect(lambda: self.Position_Subtitles_Set('ne'))
        self.sub_pos_E_Button.clicked.connect(lambda: self.Position_Subtitles_Set('e'))
        self.sub_pos_SE_Button.clicked.connect(lambda: self.Position_Subtitles_Set('se'))
        self.sub_pos_S_Button.clicked.connect(lambda: self.Position_Subtitles_Set('s'))
        self.sub_pos_SW_Button.clicked.connect(lambda: self.Position_Subtitles_Set('sw'))
        self.sub_pos_W_Button.clicked.connect(lambda: self.Position_Subtitles_Set('w'))
        self.sub_pos_C_Button.clicked.connect(lambda: self.Position_Subtitles_Set('c'))

        # "Субтитры" для бота - subtitles for bot
        self.thread = None
        self.Sub_Button.clicked.connect(self.on_btn)

        self.Sub_Button_icon_on = QtGui.QIcon('icon/subtitles_on.png')
        self.Sub_Button_icon_off = QtGui.QIcon('icon/subtitles_off.png')
        self.Sub_Button.setIcon(self.Sub_Button_icon_off)
        self.Sub_Button.setIconSize(QtCore.QSize(24, 24))

        self.user_input_signal_history.connect(self.handle_user_input_history)
        self.user_input_signal.connect(self.handle_user_input)

    def handle_user_input_history(self,text):
        print(text)
        self.History_Sett_TE.setText(History_Mem_CP)
        self.Save_OpenAI_settings()

    def handle_user_input(self, text):
        global Chat_message_TG_bot
        self.out_text = ' '.join(Replacing_The_Text(text))
        print(self.out_text)
        self.Telegram_bot_view.append(Chat_message_TG_bot)
        self.Telegram_bot_view.moveCursor(self.Telegram_bot_view.textCursor().End)
        try:
            self.conv_text_to_func()
        except Exception as f:
            print(f)

    def Save_tgAPI(self):
        api_key_tg = self.settings_apikey_Tg.text()
        with open("Telegram_API.txt", "w", encoding="utf-8") as file:
            file.write(api_key_tg)


    def Position_Subtitles_Set(self,pos):
        self.SubPos_Save(pos)

    def SubPos_Save(self,pos=None):
        global X_axis, Y_axis,Sub_Pos
        try:
            screen = QApplication.primaryScreen()
            screen_size = screen.size()
            X_size_screen_monitor = screen_size.width()
            Y_size_screen_monitor = screen_size.height()
            Position_Sub_dict = {'nw': [0, 0],
                                 'n': [X_size_screen_monitor // 2 - 250, 0],
                                 'ne': [X_size_screen_monitor - 500, 0],
                                 'w': [0, Y_size_screen_monitor // 2 - 50],
                                 'c': [X_size_screen_monitor // 2 - 250, Y_size_screen_monitor // 2 - 50],
                                 'e': [X_size_screen_monitor - 500, Y_size_screen_monitor // 2 - 50],
                                 'sw': [0, Y_size_screen_monitor - 100],
                                 's': [X_size_screen_monitor // 2 - 250, Y_size_screen_monitor - 100],
                                 'se': [X_size_screen_monitor - 500, Y_size_screen_monitor - 100]}
            Position_Sub_ButtonSet = {'nw': self.sub_pos_NW_Button,
                                 'n': self.sub_pos_N_Button,
                                 'ne': self.sub_pos_NE_Button,
                                 'w':  self.sub_pos_W_Button,
                                 'c':  self.sub_pos_C_Button,
                                 'e':  self.sub_pos_E_Button,
                                 'sw': self.sub_pos_SW_Button,
                                 's':  self.sub_pos_S_Button,
                                 'se': self.sub_pos_SE_Button}
            for btn in Position_Sub_ButtonSet.values():
                btn.setStyleSheet(self.Style_Sheet_norm)

            if pos in Position_Sub_dict:
                Position_Sub_ButtonSet[pos].setStyleSheet(self.Style_Sheet_act)

            X_axis = self.Add_to_Xaxis_LE.text()
            Y_axis = self.Add_to_Yaxis_LE.text()

            if X_axis =='' or not X_axis.isdigit():
                self.Add_to_Xaxis_LE.setText('0')
                X_axis = 0
            else:
                X_axis = int(X_axis)

            if Y_axis == '' or not Y_axis.isdigit():
                self.Add_to_Yaxis_LE.setText('0')
                Y_axis = 0
            else:
                Y_axis = int(Y_axis)
            if pos in Position_Sub_dict:
                Sub_Pos = Position_Sub_dict[pos]
            Sub_Pos[0] = Sub_Pos[0] + X_axis
            Sub_Pos[1] = Sub_Pos[1] + Y_axis
            parameter_save_file['Sub_Pos']=pos
            parameter_save_file['Sub_Pos_XY'] = Sub_Pos
            parameter_save_file['Xaxis'] = X_axis
            parameter_save_file['Yaxis'] = Y_axis
            self.save_savefile()
            if self.thread:
                self.SubWin.close()
                self.SubWin = Sub_Win(self.thread)
                self.SubWin.show()
        except Exception as f:
            print(f)
        print('> Sub position -',pos, Sub_Pos)

    def ClearList_SitesProg_side(self,object):
        if self.animation_block:
            return
        self.animation_block = True

        obj_set = {self.Confirmation_clearing_site_list: lambda:setattr(self,'clear_site_list_confirm', 0),
                   self.Confirmation_clearing_prog_list: lambda:setattr(self,'clear_prog_list_confirm', 0)}
        obj_set[object]()

        self.animation1 = QtCore.QPropertyAnimation(object, b"maximumWidth")
        self.animation1.setDuration(self.duration_anim_sideMenu)
        self.animation1.setStartValue(225)
        self.animation1.setEndValue(75)
        self.animation1.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        self.animation1.finished.connect(self.on_animation_finished)
        self.animation1.start()

    def ClearList_SitesProg_side_in(self,object):
        if self.animation_block:
            return
        self.animation_block = True

        obj_set = {self.Confirmation_clearing_site_list: lambda:setattr(self,'clear_site_list_confirm', 1),
                   self.Confirmation_clearing_prog_list: lambda:setattr(self,'clear_prog_list_confirm', 1)}
        obj_set[object]()
        print(object)


        self.animation10 = QtCore.QPropertyAnimation(object, b"maximumWidth")
        self.animation10.setDuration(self.duration_anim_sideMenu)
        self.animation10.setStartValue(75)
        self.animation10.setEndValue(225)
        self.animation10.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        self.animation10.finished.connect(self.on_animation_finished)
        self.animation10.start()

    def ClearList_Sites(self,Class):
        global Saved_Sites,parameter_save_file,Saved_Prog

        def Site_Clear():
            if self.clear_site_list_confirm:
                self.Browser_saved_sites.clear()
                Saved_Sites = []
                parameter_save_file['Saved_Sites'] = Saved_Sites
                self.ClearList_SitesProg_side(self.Confirmation_clearing_site_list)
                self.save_savefile()
                print(self.clear_site_list_confirm)
            if self.clear_site_list_confirm == 0:
                self.ClearList_SitesProg_side_in(self.Confirmation_clearing_site_list)

        def Prog_Clear():
            if self.clear_prog_list_confirm:
                self.Browser_saved_prog.clear()
                Saved_Prog = []
                parameter_save_file['Saved_Prog'] = Saved_Prog
                self.ClearList_SitesProg_side(self.Confirmation_clearing_prog_list)
                self.save_savefile()
                print(self.clear_prog_list_confirm)
            if self.clear_prog_list_confirm == 0:
                self.ClearList_SitesProg_side_in(self.Confirmation_clearing_prog_list)

        set_def_Clear = {"Site":lambda : Site_Clear(),
                         "Prog":lambda : Prog_Clear()}

        set_def_Clear[Class]()

    def Save_Sites_Adress_Name(self,Code_Name): # сохранение имени и адреса страницы или сайта
        global Saved_Sites,parameter_save_file , Saved_Prog
        print(Code_Name)
        try:
            if Code_Name == "Site":
                Name_and_adress = {''.join(Replacing_The_Text(self.Name_sites_LE.text())):self.Adress_sites_LE.text()}
                if Name_and_adress not in Saved_Sites and Name_and_adress !={'':''}: # отсутствие допуска уже введенных переменных
                    Saved_Sites.append(Name_and_adress)
                    parameter_save_file['Saved_Sites'] = Saved_Sites
                    self.save_savefile()

                self.Browser_saved_sites.clear()
                self.Name_sites_LE.clear()
                self.Adress_sites_LE.clear()

                self.Conv_sites_to_text()
                self.Browser_saved_sites.setText(self.saved_sites_text_full)

            else:
                Name_and_adress = {self.Name_prog_LE.text(): self.Adress_prog_LE.text()}
                if Name_and_adress not in Saved_Prog and Name_and_adress !={'':''}:  # отсутствие допуска уже введенных переменных
                    Saved_Prog.append(Name_and_adress)
                    parameter_save_file['Saved_Prog'] = Saved_Prog
                    self.save_savefile()
                print("progi",Saved_Prog)

                self.Browser_saved_prog.clear()
                self.Name_prog_LE.clear()
                self.Adress_prog_LE.clear()

                self.Conv_prog_to_text()
                self.Browser_saved_prog.setText(self.saved_prog_text_full)
        except Exception as f:
            print(f)

    def Conv_sites_to_text(self):
        print("текст из сайтов")
        try:
            self.conv_name_sites = {}
            saved_sites_text=[]
            for string in Saved_Sites:
                for i, a in string.items():
                    if i and a:
                        if not a.startswith('https://'):
                            a = 'https://' + a
                        saved_sites_text.append(f'{i}  |  {a}')
                        self.conv_name_sites[i] = a

            self.saved_sites_text_full = '\n'.join(saved_sites_text)
            self.Browser_saved_sites.setText(self.saved_sites_text_full)
        except Exception as f:
            print(f)

    def Conv_prog_to_text(self):
        self.conv_name_prog = {}
        saved_prog_text = []
        for string in Saved_Prog:
            for i, a in string.items():
                if i and a:
                    saved_prog_text.append(f'{i}  |  {a}')
                    self.conv_name_prog[i]=a


        self.saved_prog_text_full = '\n'.join(saved_prog_text)
        self.Browser_saved_prog.setText(self.saved_prog_text_full)

    def PathTo_setAdressProg(self):
        file_dialog = QFileDialog()
        file_path, _ = file_dialog.getOpenFileName()
        self.Adress_prog_LE.setText(file_path)

    def Send_Message_Sub(self,response):
        if self.thread:
            self.thread.send_param.emit(response)

    def on_btn(self):
        if self.thread is None:
            self.Sub_Button.setIcon(self.Sub_Button_icon_on)
            self.thread = ThreadWindow()
            self.thread.threadSignal.connect(self.on_threadSignal)
            self.thread.start()

            self.SubWin = Sub_Win(self.thread)
            self.SubWin.show()
        else:
            self.Sub_Button.setIcon(self.Sub_Button_icon_off)
            self.thread.stop()
            self.thread.quit()
            self.thread.wait()
            self.thread = None

            self.SubWin.close()

    def on_threadSignal(self, value):
        print(f"Received from thread: {value}")

    def Change_Voice_Speaker(self,speaker):
        global Speaker_Voice
        Style_Sheet_act = self.Style_Sheet_act
        Style_Sheet_norm = self.Style_Sheet_norm

        buttons = {
            'baya': self.baya_tButton,
            'kseniya': self.kseniya_tButton,
            'xenia': self.xenia_tButton,
            'eugene': self.eugene_tButton,
            'aidar': self.aidar_tButton
        }

        for btn in buttons.values():
            btn.setStyleSheet(Style_Sheet_norm)

        if speaker in buttons:
            buttons[speaker].setStyleSheet(Style_Sheet_act)

        parameter_save_file['Speaker_Voice'] = speaker
        Speaker_Voice= speaker
        self.save_savefile()

    def off_or_on_sint_voice(self,checked):
        if checked:
            print('> voice is turned on')
            self.deactivate_voice.setIcon(self.icon_voice_active)
            self.check_activate_sint_voice=True
        else:
            print('> voice is turned off')
            self.deactivate_voice.setIcon(self.icon_voice_normal)
            self.check_activate_sint_voice = False

    def fifteens_times_history_gen(self):
        global History_Mem_CP
        chat_history = self.memory.get_messages()
        chat_string = "\n".join(f"{entry['role']}: {entry['content']}" for entry in chat_history)
        text = ('выдели информацию из всего текста (в частности у пользователя) такую что может пригодиться на долгое время, что можно вcпомнить на будущее.'
                'и то что попросил пользователь у ассистента. сделать все очень кратко по возможности в одного предложения. пиши словно идет обращение к ассистенту. при отсутствии важной инфорvации вывести пустоту.'
                'нужен только текст о том какой ассистент должен быть с тем, что должен помнить')
        sys_message  =f"'{chat_string}' - {text}"
        messages_with_system = [{"role": "system", "content": sys_message}]
        response = generate_response_history(messages_with_system)
        print(">>>",response, "<<<")
        messages_with_system = [{"role": "system", "content": (f"'{response}' - добавь текст из первого текста в этот '{History_Mem_CP}' адаптируя первый текст под стиль написании второго, но не меняя второй текст просто добавь из первого. пиши словно идет обращение к ассистенту. текст должен быть написано к ассистенту."
                                                               f"должно быть все написано обезличено без упоминания пользвателя и ассистента. выведи только в один текст.")}]
        response = generate_response_history(messages_with_system)
        print("<<<",response,">>>" )
        History_Mem_CP = response
        parameter_save_file['History_Mem_CP'] = History_Mem_CP
        self.History_Sett_TE.setText(History_Mem_CP)
        self.Save_OpenAI_settings()

    def timer_interval_set(self):
        self.interval = random.randint(Min_STime_Question*1000, Max_STime_Question*1000)

        print(f'{self.interval//1000} sec. ({(self.interval//1000)//60} min. {(self.interval//1000)%60} sec.)  left before the question')

    def execute_command(self):
        print("> Random question!")

        user_input = ("*задай краткий любой глупый вопрос или скажи что-нибуль утвердительнон НО не вопрос. связанный с историей чата и его темой, как мой друг,"
                      "обязательно не повторяйся в вопросах. совмещай утверждения или говори только про одно. пиши краткр в одно предложение максимум! если до тебе не ответили, то веди себя агрессивно и молчи в ответ*")
        self.memory.add_message("user", user_input)
        messages_with_system = [{"role": "system", "content": self.system_message}] + self.memory.get_messages()
        try:
            response = generate_response(messages_with_system)
            self.memory.add_message("assistant", response)
            self.textBrowser.append(f"Bot: {response}")
            self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
            self.history_edit_check +=1
            if self.history_edit_check >= 15:
                print(self.history_edit_check)
                self.fifteens_times_history_gen()
                self.history_edit_check = 0
            print(response)
            self.Send_Message_Sub(response)
            if self.check_activate_sint_voice:
                self.voice_massage_ask(response)
        except Exception as error:
            print(f'error - {error}')

        parameter_save_file["Chat_history"] = self.memory.get_messages()

        print(self.memory, parameter_save_file)
        self.save_savefile()

        self.timer_interval_set()
        self.timer.start(self.interval)

    def voice_adoptation(self):
        self.synthesize_and_play('о')
        print('> voice synthesis loading, test:', 1)
        self.synthesize_and_play('завершён...')
        print('> voice synthesis loading is complete...')

    def load_save_file(self): # Initialization/Инициализация

        last_backslash_index = self.file_path.rfind('/')
        last_part = self.file_path[last_backslash_index + 1:]
        if last_part in self.browsers_executables:
            self.temp_browser_set = self.file_path
            print('> загрузка .exe файла браузера temp_browser_set:', self.temp_browser_set)

        # set info API tokens and temperature
        self.Num_tokens_LineE.setText(str(Number_of_tokens))
        self.Temperature_LineE.setText(str(Temperature))

        # set info min/max question interval
        self.Question_Interval_min_LE.setText(str(Min_STime_Question))
        self.Question_Interval_max_LE.setText(str(Max_STime_Question))

        # set Prompt's - Personality, History, Manner speech
        self.Personality_Sett_TE.setText(Personality_CP)
        self.History_Sett_TE.setText(History_Mem_CP)
        self.Manner_Sett_TE.setText(Manner_Voice_CP)
        print(f'> system prompts - {self.system_message}')

        self.Change_Voice_Speaker(Speaker_Voice)

        # set API for TgBot
        self.settings_apikey_Tg.setText(TG_API)

        self.Conv_prog_to_text()
        self.Conv_sites_to_text()

        self.Browser_saved_sites.setText(self.saved_sites_text_full)

        # Subtitles position
        self.Add_to_Xaxis_LE.setText(str(X_axis))
        self.Add_to_Yaxis_LE.setText(str(Y_axis))
        self.SubPos_Save(Sub_Pos_W)

        """self.voice_adoptation()"""
        self.click_browser_1.setText(self.file_path)
        print('> loading API key...')
        if OpenAi_ApiKey:
            self.settings_apikey.setText(OpenAi_ApiKey)
            print('> loading API key successful')
        else:
            self.textBrowser.setText('loading API key unsuccessful')

    def Save_Ttaaq(self): # save Timer
        global Min_STime_Question,Max_STime_Question
        parameter_save_file['Question_Interval_min'] = int(self.Question_Interval_min_LE.text())
        parameter_save_file['Question_Interval_max'] = int(self.Question_Interval_max_LE.text())
        Min_STime_Question = int(self.Question_Interval_min_LE.text())
        Max_STime_Question = int(self.Question_Interval_max_LE.text())
        self.save_savefile()
        self.timer_interval_set()
        self.timer.start(self.interval)

    def Save_OpenAI_settings(self):
        global Manner_Voice_CP,History_Mem_CP,Personality_CP
        try:
            api_key_text = self.settings_apikey.text()
            Manner_Voice_CP = str(self.Manner_Sett_TE.toPlainText())
            History_Mem_CP = str(self.History_Sett_TE.toPlainText())
            Personality_CP = str(self.Personality_Sett_TE.toPlainText())
            parameter_save_file['Number_of_tokens']=int(self.Num_tokens_LineE.text())
            parameter_save_file['Temperature'] = float(self.Temperature_LineE.text())
            parameter_save_file['Manner_Voice_CP'] = Manner_Voice_CP
            parameter_save_file['History_Mem_CP'] = History_Mem_CP
            parameter_save_file['Personality_CP'] = Personality_CP
            self.save_savefile()

            with open("OpenAiApiKey.txt", "w", encoding="utf-8") as file:
                file.write(api_key_text)
        except Exception as f:
            print(f)

    def Set_Default_Settings(self):
        self.Num_tokens_LineE.setText(str(parameter_save_file['Number_of_tokens']))
        self.Temperature_LineE.setText(str(parameter_save_file['Temperature']))

        self.Question_Interval_min_LE.setText(str(parameter_save_file['Question_Interval_min']))
        self.Question_Interval_max_LE.setText(str(parameter_save_file['Question_Interval_max']))

    def handle_input(self):
        user_input = self.lineEdit_2.text()
        try:
            if user_input:
                self.memory.add_message("user", user_input)
                messages_with_system = [{"role": "system", "content": self.system_message}] + self.memory.get_messages()
                try:
                    response = generate_response(messages_with_system)
                    self.memory.add_message("assistant", response)
                    self.textBrowser.append(f"User: {user_input}")
                    self.textBrowser.append(f"Bob: {response}")

                    self.history_edit_check += 1
                    if self.history_edit_check == 10:
                        print(self.history_edit_check)
                        self.fifteens_times_history_gen()
                        self.history_edit_check = 0
                    print(response)
                    self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
                except Exception as error:
                    print(f'error - {error}')
                    return

                self.lineEdit_2.clear()

                self.Send_Message_Sub(response)

                if self.check_activate_sint_voice:
                    self.voice_massage_ask(response)
                self.out_text = " ".join(Replacing_The_Text(user_input))
                self.conv_text_to_func()

                parameter_save_file["Chat_history"]=self.memory.get_messages()
                print(self.memory,parameter_save_file)
                self.save_savefile()
            self.timer_interval_set()
            self.timer.start(self.interval)


        except Exception as p:
            print(f'Error: {p}')

    def open_browser_file(self):
        file_dialog = QFileDialog()
        self.browser_var = ['']
        self.file_path, _ = file_dialog.getOpenFileName()

        self.load_save_file()

        parameter_save_file["Browser_directory"] = self.file_path
        self.save_savefile()
        self.default_browser_state = 1
        self.voice_massage_ask(
            f"{self.dict_word_used_browser[random.randint(0, len(self.dict_word_used_browser) - 1)]}"
            f" {self.file_path}"
        )

    def save_savefile(self):
        with open('savefile.json', 'w',encoding='utf-8') as json_file:
            json.dump(parameter_save_file, json_file,ensure_ascii=False,indent=len(parameter_save_file))
        print('savefile.json saved in folder')

    def delete_save_browser_default(self):
        self.click_browser_1.setText('None')
        self.file_path = None
        print('the path to the browsers .exe file', self.file_path)
        self.default_browser_state = 0


    def qwe(self):
        def get_random_message():
            # Выбираем случайное предложение из списка
            message = random.choice(synonyms)

            # Список дополнительных фраз
            additional_phrases = [
                "Прошу удалить этот пункт",
                "Удалите, пожалуйста, данный элемент",
                "Будьте добры убрать этот пункт",
                "Просьба удалить указанный пункт",
                "Пожалуйста, исключите данный пункт",
                "Уберите, пожалуйста, этот пункт",
                "Прошу исключить указанный пункт",
                "Пожалуйста, уберите этот элемент",
                "Прошу удалить данный элемент",
                "Будьте любезны удалить этот пункт"
            ]

            # Выбираем случайное дополнительное сообщение
            additional_message = random.choice(additional_phrases)

            # Формируем окончательное сообщение
            final_message = f"{message} {additional_message}"

            return final_message
        synonyms = [
            "Браузер не обнаружен в системе.",
            "Браузер отсутствует в базе данных.",
            "Браузер не зарегистрирован в списке.",
            "Данный браузер не найден в реестре.",
            "Браузер не числится в базе.",
            "Нет информации о данном браузере в базе.",
            "Браузер не представлен в базе данных.",
            "В базе данных нет сведений о браузере.",
            "Браузер не найден в перечне.",
            "Этот браузер отсутствует в списке поддерживаемых."
        ]
        index = random.randint(0, len(synonyms) - 1)
        self.voice_massage_ask(get_random_message())

    def animations_slide(self,Properties,Object,Start_Value,End_Value):
        if self.animation_block:
            return
        self.animation_block = True

        self.animation = QtCore.QPropertyAnimation(Properties, Object)
        self.animation.setDuration(self.duration_anim_sideMenu)
        self.animation.setStartValue(Start_Value)
        self.animation.setEndValue(End_Value)
        self.animation.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        self.animation.finished.connect(self.on_animation_finished)
        self.animation.start()

    def animations_slide_1(self, Properties, Object, Start_Value, End_Value):
        if not self.anim_availability:
            if self.animation_block:
                return
            self.animation_block = True

        self.animation_1 = QtCore.QPropertyAnimation(Properties, Object)
        self.animation_1.setDuration(self.duration_anim_sideMenu)
        self.animation_1.setStartValue(Start_Value)
        self.animation_1.setEndValue(End_Value)
        self.animation_1.setEasingCurve(QtCore.QEasingCurve.InOutQuart)
        self.animation_1.finished.connect(self.on_animation_finished)
        self.animation_1.start()
        self.anim_availability = False

    def side_Tg_bot_consol(self):
        if self.TgB_Manner == 0: #открытие окна
            self.TgB_Manner = 1
            self.side_Tg_bot_Button.setText(">")
            self.anim_availability = True
            self.animations_slide(self.Telegram_bot, b"maximumWidth",10,360)
            self.animations_slide_1(self.frame_28, b"maximumHeight", 55, 120)
        else:                     #закрытие окна
            self.TgB_Manner = 0
            self.side_Tg_bot_Button.setText("<")
            self.anim_availability = True
            self.animations_slide(self.Telegram_bot, b"maximumWidth",360,10)
            self.animations_slide_1(self.frame_28, b"maximumHeight", 120, 55)


    def Anim_Slide_Frame_Manner_TE(self):
        if self.ASF_Manner_TE == 0: #открытие окна
            self.ASF_Manner_TE = 1
            self.Manner_Sett_TE_Button.setText(" < ")
            self.animations_slide(self.Manner_Sett_TE, b"minimumHeight", 0, 170)
        else:                     #закрытие окна
            self.ASF_Manner_TE = 0
            self.Manner_Sett_TE_Button.setText(". . .")
            self.animations_slide(self.Manner_Sett_TE,b"minimumHeight",170,0)


    def Anim_Slide_Frame_History_TE(self):
        if self.ASF_History_TE == 0: #открытие окна
            self.ASF_History_TE = 1
            self.History_Sett_TE_Button.setText(" < ")
            self.animations_slide(self.History_Sett_TE, b"minimumHeight", 0, 170)
        else:                     #закрытие окна
            self.ASF_History_TE = 0
            self.History_Sett_TE_Button.setText(". . .")
            self.animations_slide(self.History_Sett_TE, b"minimumHeight", 170, 0)

    def Anim_Slide_Frame_Pers_TE(self):
        if self.ASF_Pers_TE == 0: #открытие окна
            self.ASF_Pers_TE = 1
            self.Personality_Sett_TE_Button.setText(" < ")
            self.animations_slide(self.Personality_Sett_TE, b"minimumHeight", 0, 170)
        else:                     #закрытие окна
            self.ASF_Pers_TE = 0
            self.Personality_Sett_TE_Button.setText(". . .")
            self.animations_slide(self.Personality_Sett_TE, b"minimumHeight", 170, 0)

    def Slide_Frame_Func_edit(self):

        if self.Side_Menu_Num == 0 and self.Side_Menu_Num_Consol == 1 and self.Side_Menu_Num_FE ==0:
            self.Side_Menu_Num_Consol =0
            self.Side_Menu_Num_FE = 1
            self.anim_availability = True
            self.animations_slide(self.Consol,b"maximumWidth",self.frame_6_max_width,0)
            self.animations_slide_1(self.function_edit,b"maximumWidth",0,self.frame_6_max_width)

        elif self.Side_Menu_Num == 1 and self.Side_Menu_Num_Consol == 0 and self.Side_Menu_Num_FE ==0:
            self.Side_Menu_Num = 0
            self.Side_Menu_Num_FE = 1
            self.anim_availability = True
            self.animations_slide(self.frame_4,b"maximumWidth",self.frame_6_max_width,0)
            self.animations_slide_1(self.function_edit,b"maximumWidth",0,self.frame_6_max_width)

        elif self.Side_Menu_Num == 0 and self.Side_Menu_Num_Consol == 0 and self.Side_Menu_Num_FE == 1:
            self.Side_Menu_Num_Consol = 1
            self.Side_Menu_Num_FE = 0
            self.anim_availability = True
            self.animations_slide(self.function_edit,b"maximumWidth",self.frame_6_max_width,0)
            self.animations_slide_1(self.Consol,b"maximumWidth",0,self.frame_6_max_width)


    def Slide_Frame_Options(self):

        if self.Side_Menu_Num == 0 and self.Side_Menu_Num_Consol == 1 and self.Side_Menu_Num_FE == 0:
            self.Side_Menu_Num = 1
            self.Side_Menu_Num_Consol =0
            self.anim_availability = True
            self.animations_slide(self.Consol, b"maximumWidth", self.frame_6_max_width, 0)
            self.animations_slide_1(self.frame_4, b"maximumWidth", 0, self.frame_6_max_width)

            self.Set_Default_Settings()

        elif self.Side_Menu_Num == 0 and self.Side_Menu_Num_Consol ==0 and self.Side_Menu_Num_FE ==1:
            self.Side_Menu_Num = 1
            self.Side_Menu_Num_FE = 0
            self.anim_availability = True
            self.animations_slide(self.function_edit, b"maximumWidth", self.frame_6_max_width, 0)
            self.animations_slide_1(self.frame_4, b"maximumWidth", 0, self.frame_6_max_width)

        elif self.Side_Menu_Num == 1 and self.Side_Menu_Num_Consol ==0 and self.Side_Menu_Num_FE ==0:
            self.Side_Menu_Num = 0
            self.Side_Menu_Num_Consol = 1
            self.anim_availability = True
            self.animations_slide(self.frame_4, b"maximumWidth", self.frame_6_max_width, 0)
            self.animations_slide_1(self.Consol, b"maximumWidth", 0, self.frame_6_max_width)

    def Slide_Frame_Main(self):

        if self.Side_Menu_Num_2 == 0:
            self.animations_slide(self.Menu, b"maximumWidth", 80, 160)
            self.Side_Menu_Num_2 = 1
        else:
            self.animations_slide(self.Menu, b"maximumWidth", 160, 80)
            self.Side_Menu_Num_2 = 0

    def on_animation_finished(self):
        self.animation_block = False

    def voice_massage_ask(self, massage):

        self.synthesize_and_play(massage,Speaker_Voice)


    def synthesize_and_play(self,text, speaker='baya', sample_rate=24000):

        text = text+'....ъъъъ'
        audio_path = self.model.save_wav(text=text, speaker=speaker, sample_rate=sample_rate)

        audio, sr = sf.read(audio_path)

        sd.play(audio, samplerate=sr)
        sd.wait()
        self.timer_interval_set()
        self.timer.start(self.interval)


    def set_volume(self):
        global temp_vol
        o_t = self.out_text
        volume=''
        conv_num = {
            'ноль':0,'один': 1, 'два': 1, 'три': 2, 'четыре': 2, 'пять': 3, 'шесть': 3, 'семь': 4, 'восемь': 4, 'девять': 5,
            'десять': 5,
            'одиннадцать': 6, 'двенадцать': 6, 'тринадцать': 7, 'четырнадцать': 7, 'пятнадцать': 8,
            'шестнадцать': 8, 'семнадцать': 9, 'восемнадцать': 9, 'девятнадцать': 10, 'двадцать': 10,
            'двадцать один': 11, 'двадцать два': 11, 'двадцать три': 12, 'двадцать четыре': 12, 'двадцать пять': 13,
            'двадцать шесть': 13, 'двадцать семь': 14, 'двадцать восемь': 14, 'двадцать девять': 15, 'тридцать': 15,
            'тридцать один': 16, 'тридцать два': 16, 'тридцать три': 17, 'тридцать четыре': 17, 'тридцать пять': 18,
            'тридцать шесть': 18, 'тридцать семь': 19, 'тридцать восемь': 19, 'тридцать девять': 20, 'сорок': 20,
            'сорок один': 21, 'сорок два': 21, 'сорок три': 22, 'сорок четыре': 22, 'сорок пять': 23,
            'сорок шесть': 23, 'сорок семь': 24, 'сорок восемь': 24, 'сорок девять': 25, 'пятьдесят': 25,
            'пятьдесят один': 31, 'пятьдесят два': 31, 'пятьдесят три': 32, 'пятьдесят четыре': 32,
            'пятьдесят пять': 33, 'пятьдесят шесть': 33, 'пятьдесят семь': 34, 'пятьдесят восемь': 34,
            'пятьдесят девять': 35, 'шестьдесят': 30,
            'шестьдесят один': 30, 'шестьдесят два': 31, 'шестьдесят три': 31, 'шестьдесят четыре': 32,
            'шестьдесят пять': 32, 'шестьдесят шесть': 33, 'шестьдесят семь': 33, 'шестьдесят восемь': 34,
            'шестьдесят девять': 34, 'семьдесят': 35,
            'семьдесят один': 35, 'семьдесят два': 36, 'семьдесят три': 36, 'семьдесят четыре': 37,
            'семьдесят пять': 37, 'семьдесят шесть': 38, 'семьдесят семь': 38, 'семьдесят восемь': 39,
            'семьдесят девять': 39, 'восемьдесят': 40,
            'восемьдесят один': 40, 'восемьдесят два': 41, 'восемьдесят три': 41, 'восемьдесят четыре': 42,
            'восемьдесят пять': 42, 'восемьдесят шесть': 43, 'восемьдесят семь': 43, 'восемьдесят восемь': 44,
            'восемьдесят девять': 44, 'девяносто': 45,
            'девяносто один': 45, 'девяносто два': 46, 'девяносто три': 46, 'девяносто четыре': 47,
            'девяносто пять': 47, 'девяносто шесть': 48, 'девяносто семь': 48, 'девяносто восемь': 49,
            'девяносто девять': 49, 'сто': 50
        }
        num = list(range(101))
        list_o_t = o_t.split()+['']+['']

        try:
            i = list_o_t.index('на') + 1
            if list_o_t[i] in conv_num:
                print(list_o_t[i:i + 2])
                if list_o_t[i + 1] in conv_num:
                    volume = conv_num[' '.join(list_o_t[i:i + 2])]
                    print('volume', volume)
                else:
                    volume = conv_num[list_o_t[i]]
                    print('volume', volume)
            elif int(list_o_t[i]) in num:
                print(list_o_t[i])
                volume = int(list_o_t[i])//2
                print('volume', volume)
        except Exception as e:
            print(f'123{e}')
        print(volume)
        if volume !='':
            if 'увеличь' in list_o_t or 'уменьши' in list_o_t or 'убавь' in list_o_t or 'добавь' in list_o_t :
                if 'увеличь' in list_o_t:
                    pyautogui.press('volumeup', presses=int(volume))
                    temp_vol += volume
                else:
                    pyautogui.press('volumedown', presses=int(volume))
                    temp_vol -= volume
            else:
                if temp_vol <0:
                    pyautogui.press('volumedown', presses=int(50))
                    temp_vol = volume
                    pyautogui.press('volumeup', presses=int(volume))
                elif temp_vol < volume:
                    pyautogui.press('volumeup', presses=int(volume - temp_vol))
                    temp_vol = volume
                else:
                    pyautogui.press('volumedown', presses=int(temp_vol - volume))
                    temp_vol = volume
        if temp_vol !=-1 or temp_vol >=0:
            temp_vol = abs(temp_vol)
        self.temp = 'sound_update'
        print(temp_vol)

    def process_name_update(self):
        cmd = 'WMIC PROCESS get Caption'
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        output, _ = proc.communicate()
        output = output.decode('utf-8')
        lines = output.strip().split('\n')[1:]
        process_names = [line.strip().split()[0] for line in lines]
        print(sorted(process_names))
        return process_names

    def func_search(self):   # Запросы и поиск
        o_t = self.out_text
        run_prog = self.process_name_update()
        if 'найти' in o_t or 'найди' in o_t or 'введи' in o_t or 'види' in o_t:
            if run_prog.count('browser.exe') < 3:
                try:
                    webbrowser.open('https://ya.ru')  # Открываем Yandex в стандартном браузере

                    self.voice_massage_ask('Открытие браузера для поиска')

                except Exception as e:
                    self.voice_massage_ask(f'Ошибка при открытии браузера: {e}')

            text = o_t
            text = text[text.find('поиск') + 6:].split()
            if len(text) != 0:
                text = '+'.join(text)
                t = ' '.join(text)

                self.voice_massage_ask(f'поиск по запросу "{t}"')

                final_text = "start" + " " + " https://ya.ru/search/?text=" + text
                os.system(final_text)
            self.temp = 'search_browser'
        elif 'браузер' in o_t:
            self.func_browser_use()

    def func_open(self):
        print('func_open')
        o_t = (self.out_text)
        list_o_t = o_t.split()+['']
        conv_num = self.conv_name_sites
        print(conv_num)
        conv_num1 = {'ютуб':'https://www.youtube.com/','вконтакте':'https://vk.com/','кинопоиск':'https://www.kinopoisk.ru',
                    'яндексмаркет':'https://market.yandex.ru/','ютюб':'https://www.youtube.com/'}
        conv_num_file = {"проводник":"explorer C:/"}
        conv_num_temp = {'ютуб':"opened_youtube",'проводник':'opened_explorer'}
        site=''
        try:
            i = list_o_t.index('открыть') + 1
            word = list_o_t[i]+list_o_t[i+1]
            print(word)
            if word in conv_num:
                site = conv_num[word]
                self.temp= conv_num_temp[word]
                if "ютуб" == word:
                    self.y_t_temp = None
                    self.number_range = 4
                self.voice_massage_ask(('открытие сайта', site))

            elif word in conv_num_file: # открытие проводника
                self.voice_massage_ask('открытие проводника')

                path_folder = conv_num_file[word]
                self.temp = conv_num_temp[word]
                os.system(path_folder)
                time.sleep(1)
                keyboard.press_and_release('Backspace')
                self.place_mid = 7
                self.place_sma = 2
                self.place_sma_2 = 2
            elif word in self.conv_name_prog:
                program = self.conv_name_prog[word]
                print(program)
                self.temp = 'opened_any_program'
                subprocess.Popen(program)

        except Exception as e:
            print(f'func_open - {e}')
        if site!='':
            final_text = "start" + " " + " " + site
            os.system(final_text)

    def func_browser_use(self):  # открытие Браузера
        print('func_browser_use - октрытие браузера')
        o_t = self.out_text
        data = ['включи','открыть']
        try:
            browser = self.temp_browser_set
            for data_low in data:
                if data_low in o_t:
                    os.system(f'"{browser}"')
                    self.temp = 'browser_open'
        except Exception as s:
            webbrowser.open('https://ya.ru')
            self.temp = 'browser_open'
            if self.default_browser_state ==0:
                self.voice_massage_ask('открытие браузера по-умолчанию из системы Виндовс, выбранный пользователем')
                self.default_browser_state = 1
            return
        if 'закрыть' in o_t:

            os.system(f"taskkill /im {browser} /f")
            self.voice_massage_ask('закрываю')

    def func_browser_search(self):
        o_t = self.out_text
        text = o_t
        text = text[text.find('написать') + 7:].split()
        if len(text) != 0:
            text = ' '.join(text)
        if self.temp == 'opened_youtube' and self.y_t_temp != 'searching':
            for _ in range(self.number_range):
                time.sleep(0.1)
                keyboard.press_and_release('Tab')
            self.tab_temp = 4
            self.y_t_temp = 'searching'
            print(self.y_t_temp)
        for _ in range(50):
            keyboard.press_and_release('backspace')
        print(text,'fbs')
        keyboard.write(f"{text}")
        time.sleep(0.1)
        keyboard.press_and_release('Enter')
        if self.temp == 'opened_youtube':
            self.y_t_temp = 'search_youtube'
            self.tab_temp = 1
            self.number_range = 5

    def func_browser_tab(self):
        print('вкладка')
        o_t = self.out_text
        if 'удалить' in o_t or 'закрыть' in o_t:
            keyboard.press_and_release('Ctrl + w')
        elif "открыть" in o_t:
            self.voice_massage_ask("открытие вкладки")
            webbrowser.open('https://ya.ru')


    def explorer_act(self):

        o_t = self.out_text
        print('начало')
        print('place_mid',self.place_mid)
        print('place_sma',self.place_sma)
        print('place_sma_2', self.place_sma_2)
        print(self.place_mid)
        dict = {
            'быстрый доступ': [6, 0, -1],
            "яндекс диск": [6, 1, -1],"этот компьютер": [6, 2, -1],"видео": [6, 3, -1],"документы": [6, 4, -1],
            "загрузки": [6, 5, -1],"изображения": [6, 6, -1],"музыка": [6, 7, -1],"обьемные обьекты": [6, 8, -1],
            "рабочий стол": [6, 9, -1],"диск один": [6, 10, -1],"диск два": [6, 11, -1],"диск три": [6, 12, -1],
            "диск четыре": [6, 13, -1],"сеть": [6, 14, -1],'папки': [7, 2, 0],'диски': [7, 2, 7], "рабочая папка":[7,14,22],
            'налево':[6,-1,-1],'на право':[7,-1,-1],"папку":[7,-1,-2]
        }
        replacements = {
            'зайти в': 'зайди', 'зайди в': 'зайди','перейти в': 'зайди','перейти во': 'зайди',
            'зайти во': 'зайди', 'зайди во': 'зайди', 'перейди в': "зайди", 'перейди во': "зайди",
            'рабочую папку': "рабочая папка",

            'диска':'диск','первая':'первый','вторая':'второй','четвёртая':'четвёртый',

            'первый диск': 'диск один', 'диск первый': 'диск один',
            'второй диск': 'диск два', 'диск второй': 'диск два',
            'третий диск': 'диск три', 'диск третий': 'диск три',
            'четвёртый диск': 'диск четыре', 'диск четвёртый': 'диск четыре',

            'диске':'диски',"компьютера":"компьютер","компьютеров":"компьютер",

            'музыку':"музыка",'перейти':"зайди",'права':"право",
            'перейди':"зайди", 'зайти': 'зайди',
        }

        for pattern, replacement in replacements.items():
            o_t = re.sub(r'\b' + re.escape(pattern) + r'\b', replacement, o_t)
        print(o_t)
        list_o_t = o_t.split()
        ind = list_o_t.index('зайди') + 1

        try:
            name_folder = o_t.split()[ind:ind + 2]
            word_1 = name_folder[0]
            word_2 = ' '.join(name_folder)
            tim =0.2
            if word_1 in dict:
                name_folder = (word_1)
            elif word_2 in dict:
                name_folder = (word_2)
            print(name_folder)
            list_dict = dict[name_folder]
            place_mid_act = list_dict[0]
            place_sma_act = list_dict[1]
            place_sma_2_act = list_dict[2]
            print(place_mid_act)
            print(place_sma_act)
            print(place_sma_2_act)
            if place_sma_2_act == place_sma_act==-1:
                if self.place_mid > place_mid_act:
                    print("назад placr_mid")
                    while self.place_mid != place_mid_act:
                        time.sleep(0.2)
                        keyboard.press_and_release('Shift + Tab')
                        self.place_mid -=1
                        print('place_mid_', place_mid_act)
                elif self.place_mid < place_mid_act:
                    print("вперед placr_mid")
                    while self.place_mid != place_mid_act:
                        time.sleep(0.2)
                        keyboard.press_and_release('Tab')
                        self.place_mid +=1
            else:
                if self.place_mid > place_mid_act:
                    print("назад placr_mid")
                    while self.place_mid != place_mid_act:
                        time.sleep(0.2)
                        keyboard.press_and_release('Shift + Tab')
                        self.place_mid -=1
                        print('place_mid_', place_mid_act)
                elif self.place_mid < place_mid_act:
                    print("вперед placr_mid")
                    while self.place_mid != place_mid_act:
                        time.sleep(0.2)
                        keyboard.press_and_release('Tab')
                        self.place_mid +=1
                time.sleep(0.5)
                print('place_sma...')
                if place_sma_act >= 0:
                    if self.place_sma < place_sma_act:
                        print('вниз place_sma')
                        while self.place_sma != place_sma_act:
                            time.sleep(0.01)
                            keyboard.press_and_release('Down')
                            self.place_sma +=1
                    elif self.place_sma > place_sma_act:
                        print('вверх place_sma')
                        while self.place_sma != place_sma_act:
                            time.sleep(0.01)
                            keyboard.press_and_release('Up')
                            self.place_sma -=1
                time.sleep(tim)
                if place_sma_2_act >=0:
                    if self.place_sma_2 < place_sma_2_act:
                        print('влево place_sma_2')
                        while self.place_sma_2 != place_sma_2_act:
                            time.sleep(tim)
                            keyboard.press_and_release('Right')
                            self.place_sma_2 +=1
                    elif self.place_sma_2 > place_sma_2_act:
                        print('вправо place_sma_2')
                        while self.place_sma_2 != place_sma_2_act:
                            time.sleep(tim)
                            keyboard.press_and_release('Left')
                            self.place_sma_2 -=1
                else:
                    self.place_sma_2 = 0
                    keyboard.press_and_release('Enter')


        except Exception as f:
            print(f'ошибка между перемещениями в проводнике - {f}')

        print('конец')
        print('place_mid',self.place_mid)
        print('place_sma',self.place_sma)
        print('place_sma_2', self.place_sma_2)

    def conv_text_to_func(self):  # Конвертирование запросов в функцию
        global temp_vol
        print('конвертация в команду')
        t_imp_list = self.out_text.split()
        print(t_imp_list)
        t_imp = ' '.join(t_imp_list)
        print("final -", t_imp)

        actions = {
            'поиск': self.func_search if self.temp != 'opened_youtube' else self.func_browser_use,
            'верх': lambda: self._navigate('Up', -1) if self.temp == 'opened_explorer' else None,
            'вниз': lambda: self._navigate('Down', 1) if self.temp == 'opened_explorer' else None,
            'браузер': self.func_browser_use,
            'вкладка': self.func_browser_tab,
            'введи': self.func_browser_search,
            'написать': self.func_browser_search,
            'громкость': self.set_volume,
            'закрыть программу': self._close_window,
            'зайди': self.explorer_act if self.temp == 'opened_explorer' else None,
            'назад': lambda: keyboard.press_and_release('Alt + Left') if self.temp == 'opened_explorer' else None,
            'вперёд': lambda: keyboard.press_and_release('Alt + Right') if self.temp == 'opened_explorer' else None,
            'открыть': self.func_open,
            'пропуск': lambda: keyboard.press_and_release('Tab'),
            'на': self.set_volume if self.temp == 'sound_update' else None,
            'выключи звук': lambda: pyautogui.press('volumedown', presses=int(50))
        }

        for key, action in actions.items():
            if key in t_imp:
                if callable(action):
                    action()
                break

    def _navigate(self, direction, increment):
        keyboard.press_and_release(direction)
        self.place_sma += increment
        print('place_mid', self.place_mid)
        print('place_sma', self.place_sma)
        print('place_sma_2', self.place_sma_2)

    def _close_window(self):
        if self.temp == 'browser_open':
            os.system("taskkill /im browser.exe /f")
        elif self.temp == 'opened_explorer':
            keyboard.press_and_release('Alt + F4')
        else:
            keyboard.press_and_release('Ctrl + F2')
            print("Close Program")
            time.sleep(1)
            keyboard.press_and_release('Enter')

    def input_Massage(self):
        self.out_text = self.lineEdit_2.text()
        self.word_voise_or_hand = 'User'
        '''self.input_massage_cons()'''
        self.handle_input()

    def input_massage_cons(self):
        if self.massage_text == '':
            self.massage_text = self.word_voise_or_hand +' > '+self.out_text
        elif self.massage_text != '' and self.out_text !='':
            self.massage_text = self.massage_text + '\n' + self.word_voise_or_hand + ' > ' + self.out_text
        if self.out_text !='':
            self.text_signal.emit(self.massage_text)
            self.textBrowser.moveCursor(self.textBrowser.textCursor().End)
            print(self.out_text)
            self.lineEdit_2.clear()
            self.conv_text_to_func()


    def startVoiceRecognition(self, checked):
        if checked:
            if hasattr(self, 'recognition_thread') and self.recognition_thread.is_alive():
                print("Распознавание уже запущено.")
                return
            self.pushButton_2.setIcon(self.icon_active)

            def voiceRecognition():
                model = Model('vosk-model-small-ru-0.22')
                rec = KaldiRecognizer(model, 16000)
                self.p = pyaudio.PyAudio()
                self.stream = self.p.open(format=pyaudio.paInt16,
                                          channels=1,
                                          rate=16000,
                                          input=True,
                                          frames_per_buffer=8000)
                self.stream.start_stream()

                try:
                    while True:
                        data = self.stream.read(8000, exception_on_overflow=False)
                        if rec.AcceptWaveform(data):
                            result = rec.Result()
                            self.out_text = json.loads(result)['text']
                            self.word_voise_or_hand = 'User'
                            """self.input_massage_cons()"""
                            if self.out_text!='':
                                self.lineEdit_2.setText(self.out_text)
                                self.handle_input()

                        else:
                            partial_result = rec.PartialResult()
                except KeyboardInterrupt:
                    print("Распознавание остановлено пользователем")
                finally:
                    self.stream.stop_stream()
                    self.stream.close()
                    self.p.terminate()
                    self.pushButton_2.setEnabled(True)
                    self.pushButton_2.setIcon(self.icon_normal)

            self.recognition_thread = threading.Thread(target=voiceRecognition)
            self.recognition_thread.start()
        else:
            self.stream.stop_stream()
            self.pushButton_2.setIcon(self.icon_normal)

    def updateTextBrowser(self, text):
        self.textBrowser.setText(text)


# Точка входа в приложение
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    root = Window()
    root.show()
    telegram_bot = TelegramBot(root)
    sys.exit(app.exec_())
