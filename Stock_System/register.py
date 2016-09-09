import tkinter as tk
from tkinter import ttk

# Create window for Remove
win_register = tk.Tk()
win_register.focus()
win_register.title("ENET Care")

# Title label
ttk.Label(win_register, text="Edit User Profile").grid(column=1, row=0)

# Display user data

# STANDARD TYPE
ttk.Label(win_register, text="Type: ").grid(column=0,
        row=1)
# Combobox
number = tk.StringVar()
numberChosen = ttk.Combobox(win_register, width=12, textvariable=number,
        state='readonly')
numberChosen['values'] = (1, 2, 3, 5, 8, 13) #get from database
numberChosen.grid(column=1, row=1)
numberChosen.current(0)
# Reads user type from database

# EXPIRE DATE
ttk.Label(win_register, text="Expire Date: ").grid(column=0, row=2) 
barcode = tk.StringVar() 
package_number = tk.Entry(win_register, width=12,
        textvariable=barcode).grid(column=1, row=2)


# barcode number
ttk.Label(win_register, text="barcode: ").grid(column=0, row=3)
barcode = tk.StringVar()
package_number = tk.Entry(win_register, width=12,
        textvariable=barcode).grid(column=1, row=3)


# dist. center 
ttk.Label(win_register, text="dist. center: ").grid(column=0,
        row=4)

# Combobox
number = tk.StringVar()
numberChosen = ttk.Combobox(win_register, width=12, textvariable=number,
        state='readonly')
numberChosen['values'] = (1, 2, 3, 5, 8, 13) #get from database
numberChosen.grid(column=1, row=4)
numberChosen.current(0)
# Reads dist. center from database


# Function Button Action
def RemovedSuccessfully():
        lbl_status.configure(text=barcode.get() + ' Removed')

# Adds buttons
btRegister = ttk.Button(win_register, text="Cancel", command=RemovedSuccessfully)
btRegister.grid(column=0, row=5)

btRegister = ttk.Button(win_register, text="Save", command=RemovedSuccessfully)
btRegister.grid(column=1, row=5)

# Status Message
lbl_status = ttk.Label(win_register, text="Message")
lbl_status.grid(column=1, row=8)




win_register.mainloop()





