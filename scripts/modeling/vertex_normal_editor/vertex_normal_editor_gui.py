from typing import Tuple

from PySide2 import QtWidgets, QtCore
import maya.app.general.mayaMixin as mayaMixin
import maya.api.OpenMaya as om2

from .vertex_normal_editor import VertexNormalEditor


class VertexNormalEditorGui(mayaMixin.MayaQWidgetBaseMixin, QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Vertex Normal Editor")
        self.resize(400, 300)

        self.x_value: QtWidgets.QDoubleSpinBox = None
        self.y_value: QtWidgets.QDoubleSpinBox = None
        self.z_value: QtWidgets.QDoubleSpinBox = None

        self.space_radio_group: QtWidgets.QButtonGroup = None

        self.blend_slider: QtWidgets.QSlider = None

        self.create_ui()
        self.show()

    @property
    def normal_x(self) -> float:
        return self.x_value.value()

    @property
    def normal_y(self) -> float:
        return self.y_value.value()

    @property
    def normal_z(self) -> float:
        return self.z_value.value()

    @property
    def normal_vector(self) -> om2.MVector:
        return om2.MVector(self.normal_x, self.normal_y, self.normal_z)

    def set_normal_x(self, value: float):
        self.x_value.setValue(value)

    def set_normal_y(self, value: float):
        self.y_value.setValue(value)

    def set_normal_z(self, value: float):
        self.z_value.setValue(value)

    def set_normal_vector(self, vector: om2.MVector):
        self.set_normal_x(vector.x)
        self.set_normal_y(vector.y)
        self.set_normal_z(vector.z)

    @property
    def blend_rate(self):
        return self.blend_slider.value() / 100

    @property
    def space(self) -> om2.MSpace:
        return om2.MSpace.kObject if self.space_radio_group.checkedId() == 0 else om2.MSpace.kWorld

    def create_ui(self):
        root_layout = QtWidgets.QVBoxLayout(self)
        self.setLayout(root_layout)

        # Utility UI
        self._draw_utility_ui(root_layout)

        # Space radio UI
        space_radio_layout = QtWidgets.QHBoxLayout()
        self.space_radio_group = QtWidgets.QButtonGroup()
        # create widgets
        object_radio = QtWidgets.QRadioButton("Object")
        world_radio = QtWidgets.QRadioButton("World")
        # init widgets
        self.space_radio_group.addButton(object_radio, id=0)
        self.space_radio_group.addButton(world_radio, id=1)
        object_radio.setChecked(True)
        # set layout
        space_radio_layout.addWidget(object_radio)
        space_radio_layout.addWidget(world_radio)
        root_layout.addLayout(space_radio_layout)

        # Axis slider UI
        # create widgets
        x_axis_layout, self.x_value = self._create_axis_slider("x", default_value=1.0)
        y_axis_layout, self.y_value = self._create_axis_slider("y")
        z_axis_layout, self.z_value = self._create_axis_slider("z")
        # set layout
        root_layout.addLayout(x_axis_layout)
        root_layout.addLayout(y_axis_layout)
        root_layout.addLayout(z_axis_layout)

        # Edit Normal Button
        # create widgets
        load_normal_button = QtWidgets.QPushButton("Get Normal from last Selections")
        # connect signal
        load_normal_button.clicked.connect(self.load_normal_from_last_selection)
        # set layout
        root_layout.addWidget(load_normal_button)

        # Blend
        blend_layout = QtWidgets.QHBoxLayout()
        # create widgets
        self.blend_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        blend_value = QtWidgets.QDoubleSpinBox()
        blend_button = QtWidgets.QPushButton("Blend Apply")
        # widget init
        self.blend_slider.setRange(0, 100)
        blend_value.setDecimals(2)
        blend_value.setSingleStep(0.01)
        blend_value.setRange(0, 1)
        # connect signal
        self.blend_slider.valueChanged.connect(lambda: blend_value.setValue(self.blend_slider.value() / 100))
        blend_value.valueChanged.connect(lambda: self.blend_slider.setValue(blend_value.value() * 100))
        blend_value.setValue(1.0)
        blend_button.clicked.connect(self.blend_apply)
        # set layout
        blend_layout.addWidget(self.blend_slider)
        blend_layout.addWidget(blend_value)
        blend_layout.addWidget(blend_button)
        root_layout.addLayout(blend_layout)

        # Mirror
        mirror_layout = QtWidgets.QHBoxLayout()
        # create widgets
        mirror_axis_x_check = QtWidgets.QCheckBox("X")
        mirror_axis_y_check = QtWidgets.QCheckBox("Y")
        mirror_axis_z_check = QtWidgets.QCheckBox("Z")
        mirror_apply_button = QtWidgets.QPushButton("Mirror Apply")
        mirror_help_text = QtWidgets.QLabel("※ 対象座標に複数頂点があった場合、全てに適用します。フェース頂点選択は非サポートです。")
        # init widgets
        mirror_axis_x_check.setChecked(True)
        mirror_axis_x_check.setFixedWidth(40)
        mirror_axis_y_check.setFixedWidth(40)
        mirror_axis_z_check.setFixedWidth(40)
        # connect signal
        mirror_apply_button.clicked.connect(
            lambda: self.mirror_apply(
                x=mirror_axis_x_check.isChecked(),
                y=mirror_axis_y_check.isChecked(),
                z=mirror_axis_z_check.isChecked()))
        # set layout
        mirror_layout.addWidget(mirror_axis_x_check)
        mirror_layout.addWidget(mirror_axis_y_check)
        mirror_layout.addWidget(mirror_axis_z_check)
        mirror_layout.addWidget(mirror_apply_button)
        root_layout.addLayout(mirror_layout)
        root_layout.addWidget(mirror_help_text)

    @staticmethod
    def _draw_utility_ui(parent_layout: QtWidgets.QLayout):
        # create widgets
        utility_group = QtWidgets.QGroupBox("Utility")
        utility_layout = QtWidgets.QHBoxLayout()
        display_vertex_normal_button = QtWidgets.QPushButton("Display Vertex Normal")
        normal_lock_button = QtWidgets.QPushButton("Normal Lock/Unlock")
        # set layout
        utility_layout.addWidget(display_vertex_normal_button)
        utility_layout.addWidget(normal_lock_button)
        utility_group.setLayout(utility_layout)

        # TODO: 処理

        parent_layout.addWidget(utility_group)

    @staticmethod
    def _create_axis_slider(
            axis: str,
            default_value: float = 0.0,
            value_float_decimals: int = 4) -> Tuple[QtWidgets.QHBoxLayout, QtWidgets.QDoubleSpinBox]:
        value_factor = 10 ** value_float_decimals

        layout = QtWidgets.QHBoxLayout()

        # create widgets
        label = QtWidgets.QLabel(f"{axis.upper()} : ")
        value = QtWidgets.QDoubleSpinBox()
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        minus_one_button = QtWidgets.QPushButton("-1")
        zero_button = QtWidgets.QPushButton("0")
        one_button = QtWidgets.QPushButton("1")
        reverse_button = QtWidgets.QPushButton("<>")
        # init widgets
        label.setFixedWidth(20)
        value.setDecimals(value_float_decimals)
        value.setSingleStep(1 / value_factor)
        value.setRange(-1, 1)
        slider.setRange(-value_factor, value_factor)
        slider.setPageStep(0.01)
        minus_one_button.setFixedWidth(20)
        zero_button.setFixedWidth(20)
        one_button.setFixedWidth(20)
        reverse_button.setFixedWidth(20)
        # connect signal
        value.textChanged.connect(lambda: slider.setValue(value.value() * value_factor))
        slider.valueChanged.connect(lambda: value.setValue(slider.value() / value_factor))
        minus_one_button.clicked.connect(lambda: value.setValue(-1))
        zero_button.clicked.connect(lambda: value.setValue(0))
        one_button.clicked.connect(lambda: value.setValue(1))
        reverse_button.clicked.connect(lambda: value.setValue(0 - slider.value()))
        # set layout
        layout.addWidget(label)
        layout.addWidget(slider)
        layout.addWidget(value)
        layout.addWidget(minus_one_button)
        layout.addWidget(zero_button)
        layout.addWidget(one_button)
        layout.addWidget(reverse_button)

        value.setValue(default_value)

        return layout, value

    def load_normal_from_last_selection(self):
        normal_editor = VertexNormalEditor()
        normal = normal_editor.get_normal_of_last_selection(self.space)
        if normal is not None:
            self.set_normal_vector(normal)

    def blend_apply(self):
        print("blend apply")
        normal_editor = VertexNormalEditor()
        edit_type = VertexNormalEditor.EditType.REPLACE if self.blend_rate >= 1 else VertexNormalEditor.EditType.ADD
        print(f"edit_type: {edit_type}")
        normal_editor.edit_normal(
            dest_vector=self.normal_vector,
            edit_type=edit_type,
            space=self.space,
            blend_weight=self.blend_rate)

    def mirror_apply(self, x: bool, y: bool, z: bool):
        print("mirror apply")
        normal_editor = VertexNormalEditor()
        normal_editor.apply_normal_to_symmetry_vertex(
            mirror_x=x,
            mirror_y=y,
            mirror_z=z,
            space=self.space)


def main():
    VertexNormalEditorGui()


# debug
if __name__ == "__main__":
    main()
