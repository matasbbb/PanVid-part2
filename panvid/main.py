#!/usr/bin/python
import gi.repository
from gi.repository import Gtk
from panvid.input import InputRegister
from panvid.predict import registered_methods
class MainAppWindow():
    def __init__(self):
        builder = Gtk.Builder()
        builder.add_from_file("gui/main.glade")
        builder.connect_signals(self)

        self._builder = builder

        self.input_init()
        self.register_init()

        window = builder.get_object("window1")
        window.show_all()

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
        self._registerOptions = {}
        m = self._builder.get_object("method1")
        for me in registered_methods.keys():
            m.append_text(me)

        m = self._builder.get_object("method2")
        for me in [""] + registered_methods.keys():
            m.append_text(me)

        self._register_stat = self._builder.get_object("statistic")

    def register_option_changed(self, widget):
        self._registerOptions[self.get_name(widget)] = self.get_value(widget)

    def register_frames(self, widget):
        self._register_stat.set_text("Started registering")
        print self._inputOptions
        if self._inputclass is not None:
            stream = self._inputclass(**self._inputOptions)


app = MainAppWindow()
Gtk.main()



