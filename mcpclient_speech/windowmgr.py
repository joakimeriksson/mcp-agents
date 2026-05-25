"""
colorsphere.windowmgr
~~~~~~~~~~~~~~~~~~~~~

Author: Anders Holst (anders.holst@ri.se), 2021

This is a helper-module to create simple interactive interfaces in python,
consisting of a single window with several sub-windows or widgets inside.
It takes care of event handling and dispatches the events to the appropriate
widgets.

"""

import sys
import matplotlib as mpl

# macOS cpython-standalone (used by uv) doesn't bundle Tcl/Tk, so TkAgg fails.
# Prefer the native MacOSX backend there; TkAgg elsewhere.
mpl.use("MacOSX" if sys.platform == "darwin" else "TkAgg")

import matplotlib.pyplot as plt

mpl.interactive(True)
mpl.rcParams["toolbar"] = "None"


class WindowMgr:
    def __init__(self, name, width, height, numx, numy, marg=0, dir="horizontal"):
        self.maxind = (numx, numy)
        self.dir = dir
        self.dxm = marg / width
        self.dym = marg / height
        self.dx = (1.0 - self.dxm) / self.maxind[0]
        self.dy = (1.0 - self.dym) / self.maxind[1]
        self.nextind = (0, 0)
        self.targetlist = []
        self.lastkeytarget = None
        self.lastbuttontarget = None
        self.motion_hook = []
        self.resize_hook = []
        self.close_hook = []
        self.globalkeydict = {}
        self.fig = plt.figure(name)
        self.pixpt = 72.0 / self.fig.dpi
        self.fig.set_size_inches((width / self.fig.dpi, height / self.fig.dpi))
        self.disable_default_keys()
        self.fig.canvas.mpl_connect("key_press_event", self.key_press_callback)
        self.fig.canvas.mpl_connect("key_release_event", self.key_release_callback)
        self.fig.canvas.mpl_connect("scroll_event", self.scroll_callback)
        self.fig.canvas.mpl_connect("button_press_event", self.button_press_callback)
        self.fig.canvas.mpl_connect("motion_notify_event", self.button_motion_callback)
        self.fig.canvas.mpl_connect("button_release_event", self.button_release_callback)
        self.fig.canvas.mpl_connect("resize_event", self.resize_callback)
        self.fig.canvas.mpl_connect("close_event", self.close_callback)

    def get_figure(self):
        return self.fig

    def set_background(self, rgb):
        self.fig.set_facecolor(rgb)

    def get_next_rect(self):
        (nx, ny) = self.nextind
        if nx < 0 or ny < 0:
            return False
        rect = (
            nx * self.dx + self.dxm,
            1.0 - (ny + 1) * self.dy,
            self.dx - self.dxm,
            self.dy - self.dym,
        )
        if self.dir == "vertical":
            ny += 1
            if ny >= self.maxind[1]:
                ny = 0
                nx += 1
                if nx >= self.maxind[0]:
                    nx = -1
        else:
            nx += 1
            if nx >= self.maxind[0]:
                nx = 0
                ny += 1
                if ny >= self.maxind[1]:
                    ny = -1
        self.nextind = (nx, ny)
        return rect

    def add_motion_callback(self, func):
        self.motion_hook.append(func)

    def add_resize_callback(self, func):
        self.resize_hook.append(func)

    def add_close_callback(self, func):
        self.close_hook.append(func)

    def register_target(self, rect, target):
        self.targetlist.append((rect, target))

    def unregister_target(self, target):
        self.targetlist = [(r, t) for (r, t) in self.targetlist if t is not target]

    def update_target(self, rect, target):
        for i in range(len(self.targetlist)):
            if self.targetlist[i][1] is target:
                self.targetlist[i] = (rect, target)

    def clear_targets(self):
        self.targetlist = []
        self.nextind = (0, 0)

    def get_callback_target(self, event):
        pos = self.fig.transFigure.inverted().transform((event.x, event.y))
        for (rect, target) in self.targetlist:
            if (
                pos[0] >= rect[0]
                and pos[0] < rect[0] + rect[2]
                and pos[1] >= rect[1]
                and pos[1] < rect[1] + rect[3]
            ):
                return target
        return None

    def install_key_action(self, key, func):
        self.globalkeydict[key] = func

    def key_press_callback(self, event):
        if event.key in self.globalkeydict:
            self.globalkeydict[event.key]()
        else:
            self.lastkeytarget = self.get_callback_target(event)
            if self.lastkeytarget and "key_press_event" in dir(self.lastkeytarget):
                self.lastkeytarget.key_press_event(event)

    def key_release_callback(self, event):
        # The release goes to the same target as the press
        if self.lastkeytarget and "key_release_event" in dir(self.lastkeytarget):
            self.lastkeytarget.key_release_event(event)

    def scroll_callback(self, event):
        # Scrolls are special - no release
        target = self.get_callback_target(event)
        if target and "scroll_event" in dir(target):
            target.scroll_event(event)

    def button_press_callback(self, event):
        self.lastbuttontarget = self.get_callback_target(event)
        if self.lastbuttontarget and "button_press_event" in dir(self.lastbuttontarget):
            self.lastbuttontarget.button_press_event(event)

    def button_motion_callback(self, event):
        # The motion goes to the same target as the press.
        # Only motion events while pressed, unless specific motion callback
        if self.lastbuttontarget and "motion_notify_event" in dir(self.lastbuttontarget):
            self.lastbuttontarget.motion_notify_event(event)
        elif self.motion_hook:
            for func in self.motion_hook:
                func(event)

    def button_release_callback(self, event):
        # The release goes to the same target as the press
        target = self.lastbuttontarget
        self.lastbuttontarget = None
        if target and "button_release_event" in dir(target):
            target.button_release_event(event)

    def resize_callback(self, event):
        if self.resize_hook:
            for func in self.resize_hook:
                func(event)

    def close_callback(self, event):
        if self.close_hook:
            for func in self.close_hook:
                func(event)

    def disable_default_keys(self):
        mpl.rcParams['keymap.xscale'] = []
        mpl.rcParams['keymap.yscale'] = []
        mpl.rcParams['keymap.grid'] = []
        mpl.rcParams['keymap.grid_minor'] = []
        mpl.rcParams['keymap.save'] = []

    def start_event_loop(self):
        self.add_close_callback(self.exit_event_loop)
        self.fig.canvas.start_event_loop(0)

    def exit_event_loop(self, *args):
        self.fig.canvas.stop_event_loop()
