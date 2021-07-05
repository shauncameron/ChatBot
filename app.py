from tkinter import *
from tkinter import ttk
import random
from bot import *
from _thread import start_new_thread

# The bot who's chatting

chatting = gary1
gary1.__load_tflearn_model__()

# Setup

root = Tk()
root.title('Bot Interface')
root.geometry('500x475')
root.resizable(0, 0)

# Chat Window

chat_window_frame = Frame(root)
chat_window = Text(chat_window_frame, bd=1, bg='white')
chat_window.place(x=5, y=5, height=500, width=475)
chat_window.pack()

# Chat Window Config

chat_window.config(state=DISABLED)
chat_window.tag_config('bot_output', foreground='dark orchid')
chat_window.tag_config('user_input', foreground='DeepSkyBlue3')


def insert_chat_text(text, end='\n', config=None):
    chat_window.config(state=NORMAL)
    chat_window.insert(END, text + end, config)
    chat_window.config(state=DISABLED)


def insert_chat_prefix(text, config=None):
    chat_window.config(state=NORMAL)
    chat_window.insert(END, text + ': ', config)
    chat_window.config(state=DISABLED)


def insert_bot_response(response):
    insert_chat_prefix(chatting.name, 'bot_output')
    insert_chat_text(response)


def insert_user_input(user_input):
    insert_chat_prefix('Trainer', 'user_input')
    insert_chat_text(user_input)

# Update developer log



# Chat Window Scrollbar

chat_window_scrollbar = Scrollbar(root, command=chat_window.yview)
chat_window_scrollbar.place(x=480, y=5, height=390)

# Message Window

message_window = Text(root, bd=1, bg='white', width=50, height=10)
message_window.place(x=25, y=400, width=380, height=40)

# Message Window Scrollbar

message_window_scrollbar = Scrollbar(root, command=message_window.yview)
message_window_scrollbar.place(x=5, y=400, height=40)

# Message Window Progress Bar

message_window_progress_bar = ttk.Progressbar(root, orient=HORIZONTAL, length=480, mode='determinate')
message_window_progress_bar.pack(side=BOTTOM, pady=5)

# Submit Message button

submit_button = Button(root, text='Submit', background='light blue', border=0, activebackground='powder blue',
                       width='40', height=10)
submit_button.place(x=410, y=400, width=85, height=40)


# Bot stuff

def submit_input(event):

    if event.keysym == 'Shift':

        return

    else:

        submission = message_window.get('1.0', END)
        message_window.delete('1.0', END)

        insert_user_input(submission.lstrip().rstrip())

        tag, intent = chatting.process(submission.lstrip().rstrip())
        insert_bot_response(random.choice(intent.responses))

        return 'break'

message_window.bind('<Return>', submit_input)

# File menu commands

def clear_chat():
    chat_window.delete('1.0', END)


# File Menu

file_menu = Menu(root, tearoff=False)
file_menu.add_command(label='Clear Chat', accelerator='Ctrl+Alt+c', command=clear_chat)
file_menu.add_command(label='Open Chat History', accelerator='Ctrl+Alt+h')
file_menu.add_command(label='Export Chat History', accelerator='Ctrl+Alt+s')

# View Menu

view_menu = Menu(root, tearoff=False)
view_menu.add_command(label='Change Text Colour')
view_menu.add_command(label='Change Background Colour')
view_menu.add_command(label='Change Message Box Colour')

# Bot Tinker Menu

bot_tinker_menu = Menu(root, tearoff=False)
bot_tinker_menu.add_command(label='Change Happiness')
bot_tinker_menu.add_command(label='Change Sadness')
bot_tinker_menu.add_command(label='Change Anger')
bot_tinker_menu.add_command(label='Change Fear')
bot_tinker_menu.add_command(label='Change Trust')
bot_tinker_menu.add_command(label='Change Disgust')
bot_tinker_menu.add_command(label='Change Anticipation')
bot_tinker_menu.add_command(label='Change Surprise')
bot_tinker_menu.add_separator()
bot_tinker_menu.add_command(label='Change Tendency')
bot_tinker_menu.add_command(label='Change Experiment Frequency')


# Developer Menu Commands


def developer_export_intents():
    chatting.__export_intents__()


def developer_reload_intents():
    chatting.__import_intents__()


def developer_recreate_data_pickle():
    chatting.__assert_pickle_data__()


def developer_recreate_model():
    chatting.__create_tflearn_model__()


def developer_refit_model():
    start_new_thread(chatting.__refit_tflearn_model__, (1000, ))


def developer_save_model():
    chatting.__save_tflearn_model__()


def developer_fullstack_reload():
    developer_export_intents()
    developer_reload_intents()
    developer_recreate_data_pickle()
    developer_recreate_model()
    developer_refit_model()


# Developer Menu

developer_console = bot.DeveloperConsole(root, (500, 475))
bot.Options.default_developer_console = developer_console

developer_menu = Menu(root, tearoff=False)
developer_menu.add_command(label='Fullstack Reload', accelerator='Ctrl+Alt+r', command=developer_fullstack_reload)
developer_menu.add_separator()
developer_menu.add_command(label='Export Intents', accelerator='Ctrl+e', command=developer_export_intents)
developer_menu.add_command(label='Reload Intents', accelerator='Ctrl+Alt+e', command=developer_reload_intents)
developer_menu.add_command(label='Recreate data pickle', accelerator='Ctrl+p', command=developer_recreate_data_pickle)
developer_menu.add_command(label='Recreate model', accelerator='Ctrl+r', command=developer_recreate_model)
developer_menu.add_command(label='Retrain model', accelerator='Ctrl+Alt+r', command=developer_refit_model)
developer_menu.add_command(label='Save model', accelerator='Ctrl+Shift+s', command=developer_save_model)
developer_menu.add_separator()
developer_menu.add_cascade(label='Bot Tinker', menu=bot_tinker_menu)
developer_menu.add_command(label='Show Developer Console', command=developer_console.show)

def change_frame(frame):

    developer_console.frame.pack_forget()
    chat_window_frame.pack_forget()

    frame.pack()

main_menu = Menu(root, tearoff=False)
main_menu.add_cascade(label='File', accelerator='Shift+Alt+f', menu=file_menu)
main_menu.add_cascade(label='Developer', accelerator='Shift+Alt+d', menu=developer_menu)
main_menu.add_command(label='Quit', accelerator='Shift+Alt+q', command=root.destroy)
main_menu.add_separator()
main_menu.add_command(label='Developer Console', command=lambda: change_frame(developer_console.frame))
main_menu.add_command(label='Trainer', command=lambda: change_frame(chat_window_frame))
root.config(menu=main_menu)

root.mainloop()
