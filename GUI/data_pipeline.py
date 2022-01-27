# GUI for running the complete data processing pipeline

import PySimpleGUI as sg
import os.path


session_list_column = [
    [
        sg.Text("Mouse Folder"),
        sg.In(size=(25,1), enable_events=True, key="-MOUSE-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Table(
            values=[], headings=['Name      ','preprocess','CaImAn','PC'], enable_events=True, size=(40,20), key="-SESSION-LIST-"
        )
    ],
]

processing_column = [
    [sg.Text("In here comes some input fields and processing status updates")],
    [sg.Text(size=(40,1), key="-TOUT-")],
]

layout = [
    [
        sg.Column(session_list_column),
        sg.VSeparator(),
        sg.Column(processing_column),
    ]
]

window = sg.Window("Data pipeline", layout)

while True:
    event,values = window.read()
    if event=="Exit" or event==sg.WIN_CLOSED:
        break

    if event=="-MOUSE-FOLDER-":
        folder = values["-MOUSE-FOLDER-"]
        try:
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            (f,'bla','x')
            for f in file_list
            if os.path.isdir(os.path.join(folder,f))
            and f.lower().startswith('session')
        ]
        fnames.sort()
        window["-SESSION-LIST-"].update(fnames)

    if event=="-SESSION-LIST-":
        try:
            filename = os.path.join(
                values["-MOUSE-FOLDER-"], values["-SESSION-LIST-"][0]
            )
            window["-TOUT-"].update(filename)
        except:
            pass


window.close()
