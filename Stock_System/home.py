import tkinter as tk
from tkinter import ttk

# Create window for Remove
win_home = tk.Tk()
win_home.focus()
win_home.title("ENET Care")

# Title label
ttk.Label(win_home, text="Home").grid(column=1, row=0)

# Buttons to access system pages

# Register
bt_register = ttk.Button(win_home, text="Register", command=RemovedSuccessfully)
bt_register.grid(column=0, row=1)

# Send
bt_send = ttk.Button(win_home, text="Send", command=RemovedSuccessfully)
bt_send.grid(column=1, row=1)

# Receive
bt_receive = ttk.Button(win_home, text="Receive", command=RemovedSuccessfully)
bt_receive.grid(column=2, row=1)

# View Stock
bt_view_stock = ttk.Button(win_home, text="View Stock", command=RemovedSuccessfully)
bt_view_stock.grid(column=0, row=2)

# Remove
bt_remove = ttk.Button(win_home, text="Remove", command=RemovedSuccessfully)
bt_remove.grid(column=1, row=2)

# Edit User Profile
bt_edit_user_profile = ttk.Button(win_home, text="Edit User Profile", command=RemovedSuccessfully)
bt_edit_user_profile.grid(column=2, row=2)

# Dist. Center Stock
bt_dist_center_stock = ttk.Button(win_home, text="Dist. Center Stock", command=RemovedSuccessfully)
bt_dist_center_stock.grid(column=0, row=3)

# Global Stock
bt_global_stock = ttk.Button(win_home, text="Global Stock", command=RemovedSuccessfully)
bt_global_stock.grid(column=1, row=3)

# Agent Activity
bt_agent_activity = ttk.Button(win_home, text="Agent Activity", command=RemovedSuccessfully)
bt_agent_activity.grid(column=2, row=3)

win_home.mainloop()




