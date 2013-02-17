#!/usr/bin/python
import gi.repository
import threading
from gi.repository import Gtk, Gdk, GObject
from panvid.input import InputRegister
from panvid.blend import BlendRegister
from panvid.predict import registered_methods, RegisterImagesDetect
class MainAppWindow():
    def __init__(self):
        builder = Gtk.Builder()
        builder.add_from_file("gui/main.glade")
        builder.connect_signals(self)

        self._builder = builder

        self.input_init()
        self.register_init()
        self.blend_init()
        window = builder.get_object("window1")
        window.set_title("PanVid")
        window.show_all()
        self.window = window

    def window_close(self, *args):
        Gtk.main_quit(*args)

    def input_init(self):
        self._input_register = InputRegister
        self._inputCombo = self._builder.get_object("input_type")
        self._inputBox = self._builder.get_object("input_box")
        self._inputWidget = None
        for name in self._input_register.keys():
            self._inputCombo.append_text(name)
        self.input_type_change(self._inputCombo)

    def input_type_change(self, wid):
        if wid.get_active_text() is None:
            return
        (clas, fileN) = self._input_register[wid.get_active_text()]
        tmp_builder = Gtk.Builder()
        tmp_builder.add_from_file("gui/input/" + fileN)
        tmp_builder.connect_signals(self)
        if self._inputWidget is not None:
            self._inputBox.remove(self._inputWidget)
        self._inputWidget = tmp_builder.get_object("box1")
        self._inputBox.add(self._inputWidget)
        self._inputOptions = {}
        self._inputclass = clas

    def get_value(self, widget):
        if isinstance(widget, gi.repository.Gtk.FileChooserButton):
            return widget.get_filename()
        elif isinstance(widget, gi.repository.Gtk.SpinButton):
            return widget.get_value()
        elif isinstance(widget, gi.repository.Gtk.ComboBoxText):
            return widget.get_active_text()
        elif isinstance(widget, gi.repository.Gtk.ToggleButton):
            return widget.get_mode()
        else:
            print type(widget)

    def get_name(self, widget):
        return Gtk.Buildable.get_name(widget)
    def input_option_changed(self, widget):
        self._inputOptions[self.get_name(widget)] = self.get_value(widget)

    def register_init(self):
        self._predicted = None
        self._registerOptions = {}
        m = self._builder.get_object("method1")
        for me in registered_methods.keys():
            m.append_text(me)

        m = self._builder.get_object("method2")
        for me in [""] + registered_methods.keys():
            m.append_text(me)

        self._register_stat = self._builder.get_object("statistic")
        self._register_progress = self._builder.get_object("register_progress")

    def register_option_changed(self, widget):
        self._registerOptions[self.get_name(widget)] = self.get_value(widget)

    def register_frames(self, widget):
        self._register_stat.set_text("Started registering")
        widget.set_sensitive(False)
        if self._inputclass is not None:
            stream = self._inputclass(**self._inputOptions)
            self.stream = stream
        if stream is None:
            self.error_message("Please select file")
        else:
            kargs = self._registerOptions
            if kargs.has_key("method1"):
                kargs["method"] = kargs["method1"]
                kargs.pop("method1")
            else:
                kargs["method"] = ""

            if kargs.has_key("method2") and len(kargs["method2"]) > 0:
                if len(kargs["method"]) > 0:
                    kargs["method"] += "-" + kargs["method2"]
                else:
                    kargs["method"] = kargs["method2"]
            if kargs.has_key("method2"):
                kargs.pop("method2")
            predictor = RegisterImagesDetect(stream)
            kargs["doneCB"] = lambda rez: self.done_registering(widget, rez)
            kargs["progressCB"] = self.registering_progress
            #t = threading.Thread(target=predictor.getDiff, kwargs=kargs)
            #t.start()
            self._predicted = predictor.getDiff(**kargs)
            if self._predicted is not None:
                self._register_stat.set_text("Homographies predicted")
        widget.set_sensitive(True)
    def done_registering(self, widget, rez):
        if rez is not None:
            self._predicted = rez
            self._register_stat.set_text("Homographies predicted")
        widget.set_sensitive(True)

    def registering_progress(self, of=1, curr=0, nst=1, st=1):
        self._register_progress.set_text("Stage %d/%d, Frame %d/%d"%(int(st), int(nst), int(curr),int(of)))
        self._register_progress.set_fraction(min(max(1.*curr/of,0),1))
        while Gtk.events_pending(): Gtk.main_iteration()

    def blend_init(self):
        self._blend_register = BlendRegister
        self._blendCombo = self._builder.get_object("blend_type")
        self._blendBox = self._builder.get_object("blend_box")
        self._blendWidget = None
        for name in self._blend_register.keys():
            self._blendCombo.append_text(name)
        self.blend_type_changed(self._blendCombo)

    def blend_type_changed(self, wid):
        if wid.get_active_text() is None:
            return
        (clas, fileN) = self._blend_register[wid.get_active_text()]
        if self._blendWidget is not None:
            self._blendBox.remove(self._inputWidget)
        if fileN is not None:
            tmp_builder = Gtk.Builder()
            tmp_builder.add_from_file("gui/blend/" + fileN)
            tmp_builder.connect_signals(self)
            self._blendWidget = tmp_builder.get_object("box1")
            self._blendBox.add(self._inputWidget)
        self._blendOptions = {}
        self._blendclass = clas

    def blend_option_changed(self, widget):
        self._blendOptions[self.get_name(widget)] = self.get_value(widget)

    def blend_image(self, wid, *args):
        if self._predicted is None:
            self.error_message("Please register images")
            return
        stream = self.stream.getClone()
        self.blender = self._blendclass(stream)
        self.blender.setParams(**self._blendOptions)
        self.blender.blendNextN(self._predicted)

    def preview_image(self, wid):
        self.prev_image(self.blender.getPano())

    def prev_image(self, img):
        image = Gtk.Image()
        image.set_from_pixbuf(img)
        a = Gtk.MessageDialog(self.window,
                   Gtk.DialogFlags.DESTROY_WITH_PARENT,
                   Gtk.MessageType.INFO,
                   Gtk.ButtonsType.CLOSE,
                   message)
        a.set_image(image)
        a.connect("response", lambda wid,*args: wid.destroy())
        a.run()

    def error_message(self, message=""):
        a = Gtk.MessageDialog(self.window,
                   Gtk.DialogFlags.DESTROY_WITH_PARENT,
                   Gtk.MessageType.ERROR,
                   Gtk.ButtonsType.CLOSE,
                   message)
        a.connect("response", lambda wid,*args: wid.destroy())
        a.run()

app = MainAppWindow()
GObject.threads_init(None)
Gtk.init(None)
Gtk.main()



