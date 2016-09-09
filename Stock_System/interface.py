import tkinter as tk
from tkinter import ttk

# Creates window
win = tk.Tk()
win.focus()
win.title("ENET Care")
# win.resizable(0, 0)

# Adds label "System name"
ttk.Label(win, text="ENET Care").grid(column=3, row=0)

# Label field name
ttk.Label(win, text="Name").grid(column=0, row=1)

# Adds confirmation message label
confAction = ttk.Label(win, text="")
confAction.grid(column=0, row=3)

# Function Button Action
def RegisteredSuccessfully():
        confAction.configure(text=confMessage.get() + ' Registered')

# Adds button
btRegister = ttk.Button(win, text="Register", command=RegisteredSuccessfully)
btRegister.grid(column=3, row=2)

# Adds textbox
confMessage = tk.StringVar()
packageNumber = tk.Entry(win, width=12, textvariable=confMessage).grid(column=1, row=1)

# *** Nao consegui atribuir focus() ***

# *** Erro para desabilitar novo botao ***
# btDisable = ttk.Button(win, text="Disabled")
# btDisable.configure(state='disabled')
# btDisable.grid(column=1, row=4)

# Combobox

ttk.Label(win, text="Choose a number: ").grid(column=0, row=4)
number = tk.StringVar()
numberChosen = ttk.Combobox(win, width=12, textvariable=number,
        state='readonly')
numberChosen['values'] = (1, 2, 3, 5, 8, 13)
numberChosen.grid(column=0, row=5)
numberChosen.current(0)
win.mainloop()

