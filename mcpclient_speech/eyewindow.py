import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from colorwidgets import gray, CCText, CCWidget
from windowmgr import WindowMgr

class ColorEye(CCWidget):
    def __init__(self, fig, rect, bg):
        super(ColorEye, self).__init__(fig, rect)
        self.gr0 = bg
        self.gr1 = gray(0.25)
        self.gr2 = gray(0.4)
        self.gr3 = gray(0.65)
        self.gr4 = gray(0.75)
        self.circ1 = mpl.patches.Ellipse(
            (0.5, 0.5), 0.8, 0.8, linewidth=0, edgecolor=self.gr2, facecolor=self.gr0
        )
        self.circ2 = mpl.patches.Ellipse(
            (0.5, 0.5), 0.8, 0.8, linewidth=0, edgecolor=self.gr1, fill=False
        )
        self.circ3 = mpl.patches.Ellipse(
            (0.5, 0.5), 0.8, 0.8, linewidth=0, edgecolor=self.gr3, fill=False
        )
        self.circ4 = mpl.patches.Ellipse(
            (0.5, 0.5), 0.8, 0.8, linewidth=0, edgecolor=self.gr2, fill=False)
        self.ax.add_artist(self.circ1)
        self.ax.add_artist(self.circ2)
        self.ax.add_artist(self.circ3)
        self.ax.add_artist(self.circ4)
        self.resize()

    def set_color(self, rgb):
        self.circ1.set_facecolor(rgb)

    def resize(self):
        self.size = min(self.get_width(), self.get_height())
        xoff = (1.0 - self.size/self.get_width()) / 2
        yoff = (1.0 - self.size/self.get_height()) / 2
        dpix = self.size/50
        dx = dpix/self.get_width()
        dy = dpix/self.get_height()
        self.circ1.width = 1.0 - 3*dx - 2*xoff
        self.circ1.height = 1.0 - 3*dy - 2*yoff
        for circ in [self.circ2, self.circ3, self.circ4]:
            circ.width = 1.0 - 4*dx - 2*xoff
            circ.height = 1.0 - 4*dy - 2*yoff
        self.circ2.set_center((0.5+dx/3, 0.5-dy/3))
        self.circ3.set_center((0.5-dx/3, 0.5+dy/3))
        self.circ1.set_linewidth(dpix*self.get_pixpt())
        for circ in [self.circ2, self.circ3]:
            circ.set_linewidth(dpix*self.get_pixpt())
        self.circ4.set_linewidth(dpix*self.get_pixpt() / 1.5)
        #self.txt.set_fontsize(self.size/7*self.get_pixpt())


class EyeWindow:
    def __init__(self, name, sdict, istate):
        self.width = 900
        self.height = 800
        self.statedict = sdict
        self.win = WindowMgr(name, self.width, self.height, 1, 1)
        self.bg = gray(0.5)
        self.eye = ColorEye(self.win.fig, (0.1, 0.05, 0.8, 0.8), self.bg)
        self.txt0 = CCText(self.win.fig, (0.5, 0.9), name, 1.0/20)
        self.txt1 = CCText(self.win.fig, (0.5, 0.45), "", 1.0/20)
        self.txt2 = CCText(self.win.fig, (0.5, 0.38), "", 1.0/40)
        self.cam_ax = self.win.fig.add_axes((0.78, 0.80, 0.20, 0.18))
        self.cam_ax.set_xticks([]); self.cam_ax.set_yticks([])
        for s in self.cam_ax.spines.values():
            s.set_visible(False)
        self.cam_im = self.cam_ax.imshow(np.zeros((2, 2, 3), dtype=np.uint8))
        self._pending_frame = None
        self.win.set_background(self.bg)
        self.win.register_target((0.15, 0.1, 0.7, 0.7), self)
        self.win.add_resize_callback(self.resize)
        self.win.add_close_callback(self.exit_event)
        self.set_state(istate)
        self.func1 = None
        self.func2 = None
        self.obj = None
        self.keydict = {}

    def set_button_callbacks(self, func1, func2, obj):
        self.func1 = func1
        self.func2 = func2
        self.obj = obj

    def set_exit_callback(self, func, obj):
        self.exitfunc = func
        self.exitobj = obj

    def resize(self, ev):
        self.eye.resize()
        self.txt0.resize()
        self.txt1.resize()
        self.txt2.resize()

    def exit_event(self, ev):
        if self.exitfunc:
            self.exitfunc(self.exitobj)

    def set_state(self, state):
        if state in self.statedict:
            col, txt1, txt2 = self.statedict[state]
            self.eye.set_color(col)
            self.txt1.text.set_text(txt1)
            self.txt2.text.set_text(txt2)

    def key_press_event(self, event):
        if event.key == "control":
            if self.func1:
                self.func1(event, self.obj)
        elif event.key in self.keydict and self.keydict[event.key]:
            self.keydict[event.key][0](event, self.keydict[event.key][1])
        else:
            print("Press", event.key)

    def key_release_event(self, event):
        if event.key == "control":
            if self.func2:
                self.func2(event, self.obj)
        #print("Release ", event.key)

    def button_press_event(self, event):
        if self.func1:
            self.func1(event, self.obj)

    def button_release_event(self, event):
        if self.func2:
            self.func2(event, self.obj)

    def set_camera_frame(self, frame_rgb):
        self._pending_frame = frame_rgb

    def check_events(self):
        frame = self._pending_frame
        if frame is not None:
            self._pending_frame = None
            self.cam_im.set_data(frame)
            if frame.shape[:2] != self.cam_im.get_array().shape[:2]:
                self.cam_ax.set_xlim(-0.5, frame.shape[1] - 0.5)
                self.cam_ax.set_ylim(frame.shape[0] - 0.5, -0.5)
            self.cam_ax.draw_artist(self.cam_im)
            self.win.fig.canvas.blit(self.cam_ax.bbox)
        self.win.fig.canvas.flush_events()

