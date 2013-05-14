#!/usr/bin/python
import gi.repository
import threading
from gi.repository import Gtk, Gdk, GObject
from panvid.input import InputRegister
from panvid.blend import BlendRegister
import cv2
from panvid.predict import register_methods_cont, register_methods_top
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
        self._inputCombo.set_active(0)
        self.input_type_change(self._inputCombo)

    def input_type_change(self, wid):
        if wid.get_active_text() is None:
            return
        (clas, fileN) = self._input_register[wid.get_active_text()]
        if self._inputWidget is not None:
            self._inputBox.remove(self._inputWidget)
        if fileN is not None:
            tmp_builder = Gtk.Builder()
            tmp_builder.add_from_file("gui/input/" + fileN)
            tmp_builder.connect_signals(self)
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
            return widget.get_active()
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
        for me in register_methods_cont.keys():
            m.append_text(me)
        m.set_active(2)

        m = self._builder.get_object("method2")
        for me in [""] + register_methods_cont.keys():
            m.append_text(me)
        m.set_active(1)

        m = self._builder.get_object("method3")
        for me in [""] + register_methods_cont.keys():
            m.append_text(me)
        m.set_active(1)

        self._register_progress = self._builder.get_object("register_progress")

    def register_option_changed(self, widget):
        self._registerOptions[self.get_name(widget)] = self.get_value(widget)

    def done_registering(self, widget, rez):
        if rez is not None:
            self._predicted = rez
        widget.set_sensitive(True)

    def registering_progress(self, progress):
        nst, st = progress
        self._register_progress.set_text("Frame %d/%d"%(int(st), int(nst)))
        self._register_progress.set_fraction(min(max(1.*st/nst,0),1))
        while Gtk.events_pending(): Gtk.main_iteration()

    def blend_init(self):
        self._blend_register = BlendRegister
        self._blendCombo = self._builder.get_object("blend_type")
        self._blendBox = self._builder.get_object("blend_box")
        self._blendWidget = None
        for name in self._blend_register.keys():
            self._blendCombo.append_text(name)
        self._blendCombo.set_active(0)
        self._blendOptions = {}
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
        #self._blendOptions = {}
        self._blendclass = clas

    def blend_option_changed(self, widget):
        self._blendOptions[self.get_name(widget)] = self.get_value(widget)

    def blend_image(self, widget, *args):
        widget.set_sensitive(False)
        if self._inputclass is not None:
            try:
                stream = self._inputclass(**self._inputOptions)
                self.stream = stream
            except:
                self.error_message("Please select file")
                widget.set_sensitive(True)
                return

            kargs = self._registerOptions
            if kargs.has_key("method1"):
                kargs["method"] = kargs["method1"]
                kargs.pop("method1")
            elif not kargs.has_key("method"):
                kargs["method"] = ""

            if kargs.has_key("method2") and len(kargs["method2"]) > 0:
                if len(kargs["method"]) > 0:
                    kargs["method"] += "-" + kargs["method2"]
                else:
                    kargs["method"] = kargs["method2"]
            if kargs.has_key("method2"):
                kargs.pop("method2")

            if kargs.has_key("method3") and len(kargs["method3"]) > 0:
                    kargs["jumpmethod"] = kargs["method3"]
            if kargs.has_key("method3"):
                kargs.pop("method3")
            if kargs.has_key("jumpmethod"):
                regmethod = register_methods_top["SKIP"]
            else:
                regmethod = register_methods_top["ALL"]

            predictor = regmethod(stream, 2, **kargs)
            #doneCB = lambda rez: self.done_registering(widget, rez)
            progressCB = self.registering_progress
            #t = threading.Thread(target=predictor.getDiff, kwargs=kargs)
            #t.start()
            #self._predicted = predictor.getDiff(**kargs)
            self.blender = self._blendclass(None, predictor, progressCB, **self._blendOptions )
            #self.blender.setParams(**self._blendOptions)
            try:
                self.blender.blendNextN(self._predicted)
            except:
                if self.blender._pano is not None:
                    self._register_progress.set_text("Error, part of image recovered")
                else:
                    self._register_progress.set_text("Unexpected error")
      
        widget.set_sensitive(True)

    def image_save_cb(self, wid):
        dialog = Gtk.FileChooserDialog("Save File",
                                      self.window,
                                      Gtk.FileChooserAction.SAVE,(
                                      Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                      Gtk.STOCK_SAVE, Gtk.ResponseType.ACCEPT));
        dialog.set_do_overwrite_confirmation (True);
        dialog.set_current_name ("Image.jpg");
        responce = dialog.run()
        if responce == Gtk.ResponseType.ACCEPT:
            uri = dialog.get_filename()
            image = self.blender.getPano()
            cv2.imwrite(uri, image)
        dialog.destroy()

    def stop_cb(self, wid):
        self.blender.stop = True

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



