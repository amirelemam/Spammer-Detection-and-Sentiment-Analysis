import tkinter as tk
from tkinter import ttk

# Create window for Remove
win_edit_user_profile = tk.Tk()
win_edit_user_profile.focus()
win_edit_user_profile.title("ENET Care")

# Title label
ttk.Label(win_edit_user_profile, text="Edit User Profile").grid(column=1, row=0)

# Display user data

# USER TYPE
ttk.Label(win_edit_user_profile, text="Type: ").grid(column=0,
        row=1)
#.configure(state="disabled")
type_ = tk.StringVar()
user_type = tk.Entry(win_edit_user_profile, width=12,
        textvariable=type_).grid(column=1, row=1)
 # Reads user type from database

# USERNAME
ttk.Label(win_edit_user_profile, text="Username: ").grid(column=0, row=2)
barcode = tk.StringVar()
package_number = tk.Entry(win_edit_user_profile, width=12,
        textvariable=barcode).grid(column=1, row=2)


# PASSWORD
ttk.Label(win_edit_user_profile, text="Password: ").grid(column=0, row=3)
barcode = tk.StringVar()
package_number = tk.Entry(win_edit_user_profile, width=12,
        textvariable=barcode).grid(column=1, row=3)


# CONFIRM USERNAME
ttk.Label(win_edit_user_profile, text="Confirm password: ").grid(column=0,
        row=4)

barcode = tk.StringVar()
package_number = tk.Entry(win_edit_user_profile, width=12,
        textvariable=barcode).grid(column=1, row=4)


# FULL NAME
ttk.Label(win_edit_user_profile, text="Full Name: ").grid(column=0, row=5)
barcode = tk.StringVar()
package_number = tk.Entry(win_edit_user_profile, width=12,
        textvariable=barcode).grid(column=1, row=5)


# EMAIL
ttk.Label(win_edit_user_profile, text="Email: ").grid(column=0, row=5)
barcode = tk.StringVar()
package_number = tk.Entry(win_edit_user_profile, width=12,
        textvariable=barcode).grid(column=1, row=5)


# DISTRIBUTION CENTER
ttk.Label(win_edit_user_profile, text="Dist. Center: ").grid(column=0,
        row=6)
#.configure(state="disabled")
barcode = tk.StringVar()
package_number = tk.Entry(win_edit_user_profile, width=12,
        textvariable=barcode).grid(column=1, row=6)

# Function Button Action
def RemovedSuccessfully():
        lbl_status.configure(text=barcode.get() + ' Removed')

# Adds buttons
btRegister = ttk.Button(win_edit_user_profile, text="Cancel", command=RemovedSuccessfully)
btRegister.grid(column=0, row=7)

btRegister = ttk.Button(win_edit_user_profile, text="Save", command=RemovedSuccessfully)
btRegister.grid(column=1, row=7)

# Status Message
lbl_status = ttk.Label(win_edit_user_profile, text="Message")
lbl_status.grid(column=1, row=8)

win_edit_user_profile.mainloop()




