import time
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


class VUMeter:
    """Vertical level bar: colored fill, optional peak line and threshold
    tick, a label below and a small value text above. Same information as
    the face UI's draw_audio_meter, in matplotlib form."""

    def __init__(self, fig, rect, label, bg):
        self.ax = fig.add_axes(rect, xticks=[], yticks=[])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_facecolor(gray(0.42))
        for s in self.ax.spines.values():
            s.set_color(gray(0.3))
            s.set_linewidth(0.8)
        self.fill = mpl.patches.Rectangle((0.1, 0), 0.8, 0.0, linewidth=0,
                                          facecolor=(0.1, 0.8, 0.1))
        self.ax.add_patch(self.fill)
        self.peak = self.ax.plot([0.1, 0.9], [0, 0], color="white",
                                 linewidth=1.2)[0]
        self.peak.set_visible(False)
        self.thresh = self.ax.plot([0, 1], [0, 0], color=gray(0.15),
                                   linewidth=0.8, linestyle=":")[0]
        self.thresh.set_visible(False)
        self.label = self.ax.text(0.5, -0.03, label, ha="center", va="top",
                                  fontsize=8, color=gray(0.25),
                                  transform=self.ax.transAxes)
        self.value = self.ax.text(0.5, 1.02, "", ha="center", va="bottom",
                                  fontsize=7, color=gray(0.25),
                                  transform=self.ax.transAxes)

    def set_threshold(self, level):
        self.thresh.set_ydata([level, level])
        self.thresh.set_visible(True)

    def update(self, level, color, peak=None, text=None):
        self.fill.set_height(max(0.0, min(1.0, level)))
        self.fill.set_facecolor(color)
        if peak is not None and peak > 0.02:
            self.peak.set_ydata([peak, peak])
            self.peak.set_visible(True)
        else:
            self.peak.set_visible(False)
        if text is not None:
            self.value.set_text(text)


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
        self.txt_indicator = CCText(self.win.fig, (0.05, 0.95), "", 1.0/40)
        self.camwin = None
        # VU meters on the left edge: mic level (with peak + dB) and VAD
        # probability (with threshold tick). Fed via set_audio_sources().
        self.vu_mic = VUMeter(self.win.fig, (0.025, 0.18, 0.030, 0.55),
                              "MIC", self.bg)
        self.vu_vad = VUMeter(self.win.fig, (0.070, 0.18, 0.030, 0.55),
                              "VAD", self.bg)
        # End-of-utterance countdown: fills while you are silent; at the top
        # the robot stops listening and starts processing. Blips pause it.
        self.vu_sil = VUMeter(self.win.fig, (0.115, 0.18, 0.030, 0.55),
                              "SIL", self.bg)
        # Oscilloscope (top-left, mirroring the camera thumbnail): last ~1s
        # of mic input (cyan) and TTS output (yellow), each normalized to
        # its own recent peak.
        self.scope_ax = self.win.fig.add_axes((0.02, 0.80, 0.22, 0.18),
                                              xticks=[], yticks=[])
        self.scope_ax.set_facecolor(gray(0.42))
        for s in self.scope_ax.spines.values():
            s.set_color(gray(0.3))
            s.set_linewidth(0.8)
        self._scope_n = 400
        self.scope_ax.set_xlim(0, self._scope_n - 1)
        self.scope_ax.set_ylim(-1.05, 1.05)
        x = np.arange(self._scope_n)
        self.scope_out = self.scope_ax.plot(
            x, np.zeros(self._scope_n), color=(0.95, 0.78, 0.15),
            linewidth=0.7)[0]
        self.scope_in = self.scope_ax.plot(
            x, np.zeros(self._scope_n), color=(0.15, 0.75, 0.9),
            linewidth=0.7)[0]
        self.scope_ax.text(0.02, 0.03, "in", color=(0.15, 0.75, 0.9),
                           fontsize=7, transform=self.scope_ax.transAxes)
        self.scope_ax.text(0.10, 0.03, "out", color=(0.95, 0.78, 0.15),
                           fontsize=7, transform=self.scope_ax.transAxes)
        self._audio_monitor = None
        self._voice_input = None
        self._voice_output = None
        self._scope_last_t = time.time()
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
        self.txt_indicator.resize()

    def exit_event(self, ev):
        if self.exitfunc:
            self.exitfunc(self.exitobj)

    def set_state(self, state):
        if state in self.statedict:
            col, txt1, txt2 = self.statedict[state]
            self.eye.set_color(col)
            self.txt1.text.set_text(txt1)
            self.txt2.text.set_text(txt2)

    def set_indicator(self, text):
        self.txt_indicator.text.set_text(text or "")

    def attach_camera_window(self, camwin):
        self.camwin = camwin

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

    def set_audio_sources(self, audio_monitor=None, voice_input=None,
                          voice_output=None):
        """Attach the sources for the VU meters and the scope: an
        AudioMonitor (rms/peak/waveform), a VoiceInput
        (vad_prob/vad_threshold/silence_progress) and a VoiceOutput
        (out_waveform)."""
        self._audio_monitor = audio_monitor
        self._voice_input = voice_input
        self._voice_output = voice_output
        if voice_input is not None:
            self.vu_vad.set_threshold(voice_input.vad_threshold)

    @staticmethod
    def _scope_trace(waveform, npoints):
        """Downsample a waveform to npoints and normalize to its own peak."""
        step = max(1, len(waveform) // npoints)
        y = waveform[::step][:npoints]
        if len(y) < npoints:
            y = np.pad(y, (npoints - len(y), 0))
        peak = float(np.max(np.abs(y)))
        return y / peak if peak > 0.02 else y

    def _update_meters(self):
        m = self._audio_monitor
        if m is not None:
            ref = max(m.max_seen, 0.001)
            level = min(1.0, m.rms / ref)
            peak = min(1.0, m.peak / ref)
            if level > 0.85:
                color = (0.9, 0.15, 0.15)
            elif level > 0.6:
                color = (0.9, 0.8, 0.1)
            else:
                color = (0.1, 0.8, 0.1)
            db = 20 * np.log10(m.rms + 1e-10)
            self.vu_mic.update(level, color, peak=peak, text=f"{db:.0f}")
        v = self._voice_input
        if v is not None:
            p = v.vad_prob
            active = p >= v.vad_threshold
            color = (0.1, 0.75, 0.9) if active else (0.7, 0.45, 0.1)
            self.vu_vad.update(p, color)
            sp = getattr(v, "silence_progress", 0.0)
            sil_color = (0.9, 0.3, 0.2) if sp > 0.75 else (0.55, 0.55, 0.85)
            remaining = (1.0 - sp) * v._vad_silence_ms / 1000.0
            self.vu_sil.update(sp, sil_color,
                               text=f"{remaining:.1f}" if sp > 0 else "")
        m = self._audio_monitor
        if m is not None and hasattr(m, "waveform"):
            self.scope_in.set_ydata(self._scope_trace(m.waveform, self._scope_n))
        vo = self._voice_output
        now = time.time()
        dt, self._scope_last_t = now - self._scope_last_t, now
        if vo is not None and hasattr(vo, "out_waveform"):
            # The playback callback only rolls the buffer while audio plays;
            # when idle, scroll zeros in at the same rate so finished speech
            # drifts off the display like the mic trace does.
            if not getattr(vo, "speaking", False):
                n = min(len(vo.out_waveform), int(dt * 24000))
                if n > 0:
                    vo.out_waveform = np.roll(vo.out_waveform, -n)
                    vo.out_waveform[-n:] = 0.0
            self.scope_out.set_ydata(
                self._scope_trace(vo.out_waveform, self._scope_n))

    def check_events(self):
        if self.camwin is not None:
            self.camwin.check_events()
        self._update_meters()
        self.win.fig.canvas.flush_events()


class CameraWindow:
    """A standalone window showing the annotated camera/face image.

    Mirrors the parts of EyeWindow that WindowMgr dispatches events to
    (key_press_event / key_release_event / exit_event) so keyboard shortcuts
    and close-to-quit behave the same regardless of which window has focus.
    The keydict is shared (same object) with the EyeWindow.
    """

    def __init__(self, name, width=640, height=480, keydict=None):
        self.width = width
        self.height = height
        self.win = WindowMgr(name, width, height, 1, 1)
        self.bg = gray(0.5)
        self.win.set_background(self.bg)
        self.cam_ax = self.win.fig.add_axes((0.0, 0.0, 1.0, 1.0))
        self.cam_ax.set_xticks([]); self.cam_ax.set_yticks([])
        for s in self.cam_ax.spines.values():
            s.set_visible(False)
        self.cam_im = self.cam_ax.imshow(np.zeros((2, 2, 3), dtype=np.uint8))
        self._pending_frame = None
        self._frame_shape = None
        self._sized = False
        self.keydict = keydict if keydict is not None else {}
        self.func1 = None
        self.func2 = None
        self.obj = None
        self.exitfunc = None
        self.exitobj = None
        self.win.register_target((0.0, 0.0, 1.0, 1.0), self)
        self.win.add_close_callback(self.exit_event)
        self.win.fig.canvas.draw()

    def set_exit_callback(self, func, obj):
        self.exitfunc = func
        self.exitobj = obj

    def exit_event(self, ev):
        if self.exitfunc:
            self.exitfunc(self.exitobj)

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

    def set_camera_frame(self, frame_rgb):
        self._pending_frame = frame_rgb

    def check_events(self):
        frame = self._pending_frame
        if frame is not None:
            self._pending_frame = None
            shape = frame.shape[:2]
            if shape != self._frame_shape:
                self._frame_shape = shape
                # set_extent updates the image's drawn region in data coords
                # AND set_xlim/set_ylim accordingly; set_data alone does not
                # touch the extent stored at imshow() time, so the new frame
                # would be rendered into the old 2-pixel placeholder region.
                self.cam_im.set_extent((-0.5, frame.shape[1] - 0.5,
                                        frame.shape[0] - 0.5, -0.5))
                if not self._sized:
                    self._sized = True
                    h, w = shape
                    scale = 640.0 / max(w, h)
                    dpi = self.win.fig.dpi
                    self.win.fig.set_size_inches((w * scale / dpi,
                                                  h * scale / dpi))
            self.cam_im.set_data(frame)
            self.win.fig.canvas.draw()
        self.win.fig.canvas.flush_events()

