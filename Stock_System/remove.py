import tkinter as tk
from tkinter import ttk

# Create window for Remove
win_remove = tk.Tk()
win_remove.focus()
win_remove.title("ENET Care")

# Title label
ttk.Label(win_remove, text="Remove").grid(column=1, row=0)

# Insert barcode
ttk.Label(win_remove, text="Barcode: ").grid(column=0, row=1)
barcode = tk.StringVar()
package_number = tk.Entry(win_remove, width=12, textvariable=barcode).grid(column=1, row=1)

# Function Button Action
def RemovedSuccessfully():
        lbl_status.configure(text=barcode.get() + ' Removed')

# Adds buttons
bt_register = ttk.Button(win_remove, text="Cancel", command=RemovedSuccessfully)
bt_register.grid(column=0, row=2)

bt_register = ttk.Button(win_remove, text="Register", command=RemovedSuccessfully)
bt_register.grid(column=1, row=2)

bt_register = ttk.Button(win_remove, text="Clear Fields", command=RemovedSuccessfully)
bt_register.grid(column=2, row=2)

# Status Message
lbl_status = ttk.Label(win_remove, text="Message")
lbl_status.grid(column=1, row=3)

win_remove.mainloop()



