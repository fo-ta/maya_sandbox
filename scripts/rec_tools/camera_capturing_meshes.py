import math
import sys
from typing import List, Union

import maya.api.OpenMaya as om2
import maya.cmds as cmds


class CameraCapturingMeshes:
    """
    メッシュがカメラに収まる位置ようにカメラを移動するクラス。

    Attributes:
        target_camera_name (str): カメラ名
        camera_trs_fn (om2.MFnTransform): カメラのトランスフォームノード
        camera_fn (om2.MFnCamera): カメラノード
        v_matrix (om2.MMatrix): カメラのビュー行列
        i_v_matrix (om2.MMatrix): カメラの逆ビュー行列
        h_fov_rad (float): 水平 FOV (ラジアン)
        v_fov_rad (float): 垂直 FOV (ラジアン)
        cam_space_bbox (om2.MBoundingBox): カメラ空間のバウンディングボックス
        ws_limit_top (float): カメラに収める限界座標(+y)
        ws_limit_bottom (float): カメラに収める限界座標(-y)
    """
    def __init__(
            self,
            camera_transform_name: str,
            camera_shape_name: str = None,
            ws_limit_top: float = sys.float_info.max,
            ws_limit_bottom: float = sys.float_info.min):
        """

        Args:
            camera_transform_name: カメラ(transform)
            camera_shape_name: カメラ(shape)。指定がない場合は transform の子から取得する。
            ws_limit_top: カメラに収める限界座標(+y)
            ws_limit_bottom: カメラに収める限界座標(-y)
        """
        self.ws_limit_top = ws_limit_top
        self.ws_limit_bottom = ws_limit_bottom
        self.target_camera_name = camera_transform_name
        self.cam_space_bbox = om2.MBoundingBox()
        # カメラシェイプが渡されて居ない場合、カメラトランスフォームの子から取得する
        if camera_shape_name is None:
            child_cameras = cmds.listRelatives(camera_transform_name, children=True, type="camera")
            # カメラシェイプが取得できない場合はエラー
            if not child_cameras:
                raise ValueError(f"Camera transform {camera_transform_name} has no camera shape.")
            camera_shape_name = child_cameras[0]
        # カメラの OpenMaya API 2.0 ノードを取得
        camera_list = om2.MSelectionList()
        camera_list.add(camera_transform_name)
        camera_list.add(camera_shape_name)
        self.camera_trs_fn = om2.MFnTransform(camera_list.getDagPath(0))
        self.camera_fn = om2.MFnCamera(camera_list.getDagPath(1))

        # カメラ行列を取得
        self.i_v_matrix: om2.MMatrix = self.camera_trs_fn.transformationMatrix()
        self.v_matrix: om2.MMatrix = self.i_v_matrix.inverse()
        p_matrix = om2.MMatrix(self.camera_fn.projectionMatrix())

        # FOV を取得
        p_mat_00 = p_matrix.getElement(0, 0)
        self.h_fov_rad = 2.0 * math.atan(1.0 / p_mat_00)
        p_mad_11 = p_matrix.getElement(1, 1)
        self.v_fov_rad = 2.0 * math.atan(1.0 / p_mad_11)

    def clear_bbox(self):
        """バウンディングボックスをクリアする。"""
        self.cam_space_bbox.clear()

    def expand_mesh(self, mesh: Union[om2.MFnMesh, om2.MDagPath, om2.MObject, str]):
        """
        メッシュを追加する。
        Args:
            mesh: 追加するメッシュ
        """
        var_type = type(mesh)
        if var_type is str:
            node_type = cmds.nodeType(mesh)
            if node_type == "transform":
                mesh = cmds.listRelatives(mesh, children=True, type="mesh", fullPath=True)
                if not mesh:
                    # メッシュが取得できない場合は中断
                    return
                mesh = mesh[0]
            mesh = om2.MFnMesh(om2.MGlobal.getSelectionListByName(mesh).getDagPath(0))
        elif var_type is om2.MObject:
            dag_path = om2.MDagPath.getAPathTo(mesh)
            mesh = om2.MFnMesh(dag_path)
        elif var_type is om2.MDagPath:
            mesh = om2.MFnMesh(mesh)
        elif var_type is not om2.MFnMesh:
            return

        self._expand_mesh_to_cam_space_bbox(mesh)

    def expand_meshes(self, meshes: List[Union[om2.MFnMesh, om2.MDagPath, om2.MObject, str]]):
        """
        メッシュを追加する。
        Args:
            meshes: 追加するメッシュのリスト
        """
        for mesh in meshes:
            self.expand_mesh(mesh)

    def _expand_mesh_to_cam_space_bbox(self, mesh_fn: om2.MFnMesh):
        """
        カメラ空間のバウンディングボックスを拡張する。
        Args:
            mesh_fn: 対象のメッシュ
        """
        # メッシュ頂点のワールド空間座標を取得
        points = mesh_fn.getPoints(om2.MSpace.kWorld)
        for point in points:
            # はみ出しを許容する範囲内に収める
            if point.y > self.ws_limit_top:
                point.y = self.ws_limit_top
            if point.y < self.ws_limit_bottom:
                point.y = self.ws_limit_bottom

            # メッシュ頂点をカメラ空間に変換
            point = point * self.v_matrix
            # バウンディングボックスに追加
            self.cam_space_bbox.expand(point)

    def capturing(self):
        """バウンディングボックスが収まる位置にカメラを移動する。"""
        # 計算に必要な値を準備
        half_h_fov_tan = math.tan(self.h_fov_rad / 2)
        half_v_fov_tan = math.tan(self.v_fov_rad / 2)
        half_depth = self.cam_space_bbox.depth / 2
        min_point = self.cam_space_bbox.min - self.cam_space_bbox.center
        max_point = self.cam_space_bbox.max - self.cam_space_bbox.center

        # min, max の各軸を収めるのに必要な Z 移動値を計算
        z_min_x = math.fabs(min_point.x) / half_h_fov_tan + half_depth
        z_max_x = math.fabs(max_point.x) / half_h_fov_tan - half_depth
        z_min_y = math.fabs(min_point.y) / half_v_fov_tan - half_depth
        z_max_y = math.fabs(max_point.y) / half_v_fov_tan + half_depth
        # カメラから最も遠い点を取得
        distance_from_center = max(z_min_x, z_max_x, z_min_y, z_max_y)
        # ワールド空間でのバウンディングボックスの中心位置にカメラを移動
        ws_center = self.cam_space_bbox.center * self.i_v_matrix
        self.camera_trs_fn.setTranslation(om2.MVector(ws_center), om2.MSpace.kWorld)
        # カメラをバウンディングボックスを収める位置に移動
        self.camera_trs_fn.translateBy(
            -self.camera_fn.viewDirection(om2.MSpace.kWorld) * distance_from_center,
            om2.MSpace.kWorld)
