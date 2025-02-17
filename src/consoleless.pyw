from main import HotkeyApp
import sys
import os
nullfile = open(os.devnull, 'w')
sys.stdout = open('stdout.txt', 'w')
sys.stderr = open('stderr.txt', 'w')

app = HotkeyApp()
app.run()
