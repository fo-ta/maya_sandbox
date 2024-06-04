from enum import Enum
import math
import re
import time
from typing import List, Tuple, Dict, Union, Optional

import maya.cmds as cmds
import maya.api.OpenMaya as om2


def debug_time_count_deco(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__}: {end_time - start_time} sec")
        return result
    return wrapper


class MeshVertexInfo:
    def __init__(
            self,
            vertex_index: int,
            face_index: int = None,
            selection_weight: float = 1.0):
        self.vertex_index = vertex_index
        self.face_index = face_index
        self.selection_weight = selection_weight
        self.pre_normal: Optional[om2.MVector] = None


class MeshVerticesInfo:
    def __init__(
            self,
            mesh_dag_path: om2.MDagPath):
        self.dag_path: om2.MDagPath = mesh_dag_path
        mesh_dag_path_fn: om2.MFnDagNode = om2.MFnDagNode(mesh_dag_path)
        self.transform_fn: om2.MFnTransform = om2.MFnTransform(mesh_dag_path_fn.parent(0))
        self.mesh_fn: om2.MFnMesh = om2.MFnMesh(mesh_dag_path)
        self.vertices: Dict[Optional[int], List[MeshVertexInfo]] = {}
        """フェースインデックスをキーとした頂点情報のリスト。キーが None の場合は頂点コンポーネント。"""

    def append(self, vertex_index: int, face_index: int = None, selection_weight: float = None):
        if selection_weight is None:
            selection_weight = 1.0
        vertex_info = MeshVertexInfo(
            vertex_index=vertex_index,
            face_index=face_index,
            selection_weight=selection_weight)
        if face_index in self.vertices.keys():
            self.vertices[face_index].append(vertex_info)
        else:
            self.vertices[face_index] = [vertex_info]

    def append_from_component(
            self,
            component,
            separate_face_vertex: bool = False,
            only_vertex: bool = False,
            only_face_vertex: bool = False):
        """
        コンポーネントから頂点情報を追加する。
        Args:
            component: コンポーネント
            separate_face_vertex: 頂点からフェース頂点を取得するフラグ。 only_face_vertex と同時使用不可。
            only_vertex: 頂点のみを取得するフラグ。 only_face と同時使用不可。
            only_face_vertex: フェース頂点のみを取得するフラグ。 only_vertex と同時使用不可。
        """
        # Only フラグが複数立っていた場合は処理しない
        if [only_vertex, only_face_vertex].count(True) > 1:
            return
        include_vertex = not only_face_vertex
        include_face_vertex = not only_vertex

        component_api_type = component.apiType()
        si_component_fn = om2.MFnSingleIndexedComponent(component)

        if (component_api_type == om2.MFn.kMeshVertComponent
                and include_vertex
                and not separate_face_vertex):
            # 頂点コンポーネントかつフェース頂点を分離しない場合
            # vertex_itr = om2.MItMeshVertex(self.dag_path, component)
            # while not vertex_itr.isDone():
            #     vertex_index = vertex_itr.index()
            #     vertex_itr.next()
            #
            #     self.append(vertex_index=vertex_index)
            for i in range(si_component_fn.elementCount):
                vertex_index = si_component_fn.element(i)
                # ウェイトがある場合は取得
                selection_weight = None if not si_component_fn.hasWeights else si_component_fn.weight(i).influence
                self.append(
                    vertex_index=vertex_index,
                    selection_weight=selection_weight)
        elif (
                (component_api_type == om2.MFn.kMeshVertComponent
                    and include_vertex
                    and separate_face_vertex)
                or (component_api_type == om2.MFn.kMeshVtxFaceComponent
                    and include_face_vertex)):
            # フェース頂点コンポーネントの場合、または頂点からフェース頂点を分離する場合
            vertex_itr = om2.MItMeshFaceVertex(self.dag_path, component)
            while not vertex_itr.isDone():
                vertex_index = vertex_itr.vertexId()
                f_index = vertex_itr.faceId()
                vertex_itr.next()

                # MEMO: ソフト選択の場合、頂点コンポーネントとして取得されるためウェイトの取得はしない
                self.append(vertex_index=vertex_index, face_index=f_index)

    def append_all_vertices(self):
        vertex_itr = om2.MItMeshVertex(self.dag_path)
        while not vertex_itr.isDone():
            vertex_index = vertex_itr.index()
            vertex_itr.next()

            self.append(vertex_index=vertex_index)

    def append_all_face_vertices(self):
        vertex_itr = om2.MItMeshFaceVertex(self.dag_path)
        while not vertex_itr.isDone():
            vertex_index = vertex_itr.vertexId()
            f_index = vertex_itr.faceId()
            vertex_itr.next()

            self.append(vertex_index=vertex_index, face_index=f_index)


class VertexNormalEditor:
    """
    Attributes:
    """
    def __init__(self):
        self.mesh_vertices_info_list: List[MeshVerticesInfo] = []

    class EditType(Enum):
        REPLACE = 0,
        ADD = 1,
        MULTIPLY = 2,
        BLEND = 3,

    class Space(Enum):
        OBJECT = 0,
        WORLD = 1,

    def _get_mesh_vertices_info(self, dag_path: om2.MDagPath) -> MeshVerticesInfo:
        """
        メッシュの頂点情報クラスの参照を返す。まだ指定のメッシュの頂点情報がリストアップされていない場合は新規作成してリストに追加して返す。
        Args:
            dag_path: メッシュの MDagPath

        Returns:
            頂点情報の参照
        """
        for i, mesh_vertices_info in self.mesh_vertices_info_list:
            if mesh_vertices_info.dag_path == dag_path:
                return mesh_vertices_info

        mesh_vertices_info = MeshVerticesInfo(mesh_dag_path=dag_path)
        self.mesh_vertices_info_list.append(mesh_vertices_info)
        return mesh_vertices_info

    def _get_selection_vertices(
            self,
            separate_face_vertex: bool = False,
            only_vertex: bool = False,
            only_face_vertex: bool = False):
        self.mesh_vertices_info_list.clear()

        sel_list = om2.MGlobal.getRichSelection(True).getSelection()
        for i in range(sel_list.length()):
            dag_path: om2.MDagPath
            component: om2.MObject
            dag_path, component = sel_list.getComponent(i)

            mesh_vertices_info = self._get_mesh_vertices_info(dag_path)
            mesh_vertices_info.append_from_component(
                component=component,
                separate_face_vertex=separate_face_vertex,
                only_vertex=only_vertex,
                only_face_vertex=only_face_vertex)

    @staticmethod
    def _set_vertex_normal(
            dag_path: om2.MDagPath,
            transform_fn: om2.MFnTransform,
            normal: om2.MVector,
            space: om2.MSpace,
            vertex_index: int,
            face_index: int = None):
        # cmds だとオブジェクトスペースでの適用になるため、ワールドスペースの場合には変換する
        if space == om2.MSpace.kWorld:
            normal = normal * transform_fn.transformationMatrix().inverse()
        # 頂点名を取得。 face_index 指定がある場合は vtxFace とする
        vertex_name = f"{dag_path.fullPathName()}.vtx[{vertex_index}]" \
            if face_index is None \
            else f"{dag_path.fullPathName()}.vtxFace[{vertex_index}][{face_index}]"

        # 頂点に法線を適用する。 Undo を有効にするために cmds で実行
        cmds.polyNormalPerVertex(
            vertex_name, edit=True, xyz=normal)

    def edit_normal(
            self,
            dest_vector: om2.MVector,
            edit_type: EditType,
            space: om2.MSpace = om2.MSpace.kObject,
            blend_weight: float = 1.0):
        self._get_selection_vertices()
        print(self.mesh_vertices_info_list[0].dag_path)

        # Undo チャンク開始
        cmds.undoInfo(openChunk=True)
        for mesh_vertices_info in self.mesh_vertices_info_list:
            transform_fn = mesh_vertices_info.transform_fn
            obj_matrix = transform_fn.transformationMatrix().inverse()

            mesh_fn = mesh_vertices_info.mesh_fn
            for face_index, mesh_vertex_info_list in mesh_vertices_info.vertices.items():
                for mesh_vertex_info in mesh_vertex_info_list:
                    vertex_index = mesh_vertex_info.vertex_index
                    if face_index is None:
                        normal = mesh_fn.getVertexNormal(
                            vertex_index,
                            True,
                            space)
                    else:
                        normal = mesh_fn.getFaceVertexNormal(face_index, vertex_index, space)

                    if edit_type == self.EditType.REPLACE:
                        normal = dest_vector
                    elif edit_type == self.EditType.ADD:
                        normal = normal + dest_vector
                    elif edit_type == self.EditType.MULTIPLY:
                        normal = normal * dest_vector

                    print(f"normal: {normal} ({type(normal)})")
                    normal = normal.normalize()

                    # 頂点に法線を適用する。 Undo を有効にするために cmds で実行
                    self._set_vertex_normal(
                        dag_path=mesh_vertices_info.dag_path,
                        transform_fn=transform_fn,
                        normal=normal,
                        space=space,
                        vertex_index=vertex_index,
                        face_index=face_index)
        # Undo チャンク終了
        cmds.undoInfo(closeChunk=True)

    def smooth_vertex_normal(
            self):
        self._get_selection_vertices(only_vertex=True)

        # Undo チャンク開始
        cmds.undoInfo(openChunk=True)
        for mesh_vertices_info in self.mesh_vertices_info_list:
            mesh_fn = mesh_vertices_info.mesh_fn
            for mesh_vertex_info_list in mesh_vertices_info.vertices.values():
                for mesh_vertex_info in mesh_vertex_info_list:
                    vertex_index = mesh_vertex_info.vertex_index
                    normal = self._get_connected_vertices_average_normal(
                        dag_path=mesh_vertices_info.dag_path,
                        mesh_fn=mesh_fn,
                        vertex_index=vertex_index,
                        space=om2.MSpace.kObject,
                        ignore_vertex_indexes=[vertex_index])
                    # 頂点に法線を適用する。 Undo を有効にするために cmds で実行
                    cmds.polyNormalPerVertex(
                        f"{mesh_vertices_info.dag_path.fullPathName()}.vtx[{vertex_index}]",
                        xyz=normal)
        # Undo チャンク終了
        cmds.undoInfo(closeChunk=True)

    @staticmethod
    def _get_connected_vertices_average_normal(
            dag_path: om2.MDagPath,
            mesh_fn: om2.MFnMesh,
            vertex_index: int,
            space: om2.MSpace,
            ignore_vertex_indexes: Union[int, List[int]] = None):
        # 無視インデックスリストを初期化
        if ignore_vertex_indexes is None:
            ignore_vertex_indexes = []
        elif isinstance(ignore_vertex_indexes, int):
            ignore_vertex_indexes = [ignore_vertex_indexes]

        vtx_itr = om2.MItMeshVertex(dag_path)
        vtx_itr.setIndex(vertex_index)
        connected_vertices = vtx_itr.getConnectedVertices()
        connected_faces = vtx_itr.getConnectedFaces()
        result_vector = om2.MVector()
        for connected_vertex in connected_vertices:
            if connected_vertex in ignore_vertex_indexes:
                continue

            # 隣接頂点が所属するフェースを取得
            vtx_itr.setIndex(connected_vertex)
            connected_vertex_faces = vtx_itr.getConnectedFaces()
            vertex_normal = om2.MVector()
            for connected_vertex_face_index in connected_vertex_faces:
                # フェースが対象の頂点の所属フェースに含まれていない場合はスキップ
                if connected_vertex_face_index not in connected_faces:
                    continue
                face_vertex_normal = mesh_fn.getFaceVertexNormal(
                    connected_vertex_face_index,
                    connected_vertex, space)
                result_vector += face_vertex_normal
            # フェース頂点の平均法線を結果に加算
            result_vector += vertex_normal.normalize()

        return result_vector.normalize()

    def get_normal_of_last_selection(self, space: om2.MSpace) -> Optional[om2.MVector]:
        """
        最後に選択した頂点、またはフェース頂点の法線を取得して返す。
        Returns:
            法線ベクトルを返す。対象が見つからなかった場合は None を返す。
        """
        sel_list = cmds.ls(selection=True, flatten=True, long=True)

        mesh_full_path = None
        vertex_index = -1
        face_index = None
        for sel in reversed(sel_list):
            if ".vtx[" in sel:
                match = re.match(
                    r"^(?P<mesh>.+)\.vtx\[(?P<vertex_index>\d+)\]",
                    sel)
                mesh_full_path = match.group("mesh")
                vertex_index = int(match.group("vertex_index"))
                break
            elif ".vtxFace[" in sel:
                match = re.match(
                    r"^(?P<mesh>.+)\.vtxFace\[(?P<vertex_index>\d+)\]\[(?P<face_index>\d+)\]",
                    sel)
                mesh_full_path = match.group("mesh")
                vertex_index = int(match.group("vertex_index"))
                face_index = int(match.group("face_index"))
                break

        if mesh_full_path is None:
            return

        mesh_fn = self.get_mesh_fn(mesh_full_path)

        return self.get_vertex_normal(
            mesh_fn=mesh_fn,
            vertex_index=vertex_index,
            face_index=face_index,
            space=space)

    @staticmethod
    def get_dag_path(mesh: str) -> om2.MDagPath:
        """
        メッシュの MDagPath を取得して返す。
        Args:
            mesh: メッシュのパス

        Returns:
            MDagPath
        """
        sel_list = om2.MGlobal.getSelectionListByName(mesh)
        return sel_list.getDagPath(0)

    @classmethod
    def get_mesh_fn(cls, mesh: Union[str, om2.MDagPath]) -> om2.MFnMesh:
        """
        メッシュの MFnMesh オブジェクトを取得して返す。
        Args:
            mesh: パス文字列、または MDagPath

        Returns:
            MFnMesh オブジェクト
        """
        dag_path = mesh if isinstance(mesh, om2.MDagPath) else cls.get_dag_path(mesh)
        return om2.MFnMesh(dag_path)

    @staticmethod
    def get_vertex_normal(
            mesh_fn: om2.MFnMesh,
            vertex_index: int,
            space: om2.MSpace,
            face_index: int = None):
        """
        頂点の法線を取得して返す。フェースインデックスが指定されている場合は、フェース頂点の法線を返す。
        Args:
            mesh_fn: MFnMesh オブジェクト
            vertex_index: 頂点インデックス
            face_index: フェースインデックス
            space: 空間

        Returns:
            法線ベクトル
        """
        if face_index is None:
            # 頂点の場合
            return mesh_fn.getVertexNormal(
                vertexId=vertex_index,
                angleWeighted=False,
                space=space)
            # TODO: フェース頂点法線の平均値を取得する
            
        else:
            # フェース頂点の場合
            return mesh_fn.getFaceVertexNormal(
                face_index,
                vertex_index,
                space)

    def apply_normal_to_symmetry_vertex(
            self,
            mirror_x: bool,
            mirror_y: bool,
            mirror_z: bool,
            space: om2.MSpace):
        """
        選択した頂点の法線を対称の頂点にコピーする。
        """
        self._get_selection_vertices(only_vertex=True)
        mirror_vector = om2.MVector(
            1 if not mirror_x else -1,
            1 if not mirror_y else -1,
            1 if not mirror_z else -1)

        # Undo チャンク開始
        cmds.undoInfo(openChunk=True)
        for mesh_vertices_info in self.mesh_vertices_info_list:
            dag_path = mesh_vertices_info.dag_path
            mesh_fn = mesh_vertices_info.mesh_fn
            for mesh_vertex_info_list in mesh_vertices_info.vertices.values():
                for mesh_vertex_info in mesh_vertex_info_list:
                    vertex_index = mesh_vertex_info.vertex_index
                    position = mesh_fn.getPoint(vertex_index, space)
                    normal = self.get_vertex_normal(
                        mesh_fn=mesh_fn,
                        vertex_index=vertex_index,
                        face_index=None,
                        space=space)
                    print(normal)
                    # 対称の頂点インデックスを取得
                    sym_vertex_indexes = self._get_symmetry_vertex_indexes(
                        dag_path=dag_path,
                        source_vertex_position=position,
                        mirror_vector=mirror_vector,
                        space=space)
                    # 対称の頂点に法線を適用する。 Undo を有効にするために cmds で実行
                    for sym_vertex_index in sym_vertex_indexes:
                        print(f"sym_vertex_index: {sym_vertex_index}")
                        self._set_vertex_normal(
                            dag_path=dag_path,
                            transform_fn=mesh_vertices_info.transform_fn,
                            normal=normal,
                            space=space,
                            vertex_index=sym_vertex_index)
        # Undo チャンク終了
        cmds.undoInfo(closeChunk=True)

    @staticmethod
    def _get_symmetry_vertex_indexes(
            dag_path: om2.MDagPath,
            source_vertex_position: om2.MPoint,
            mirror_vector: om2.MVector,
            space: om2.MSpace) -> List[int]:
        """
        頂点の対称の頂点インデックスを取得して返す。
        """
        # 頂点の座標を取得
        search_vertex_position = source_vertex_position
        search_vertex_position.x *= mirror_vector.x
        search_vertex_position.y *= mirror_vector.y
        search_vertex_position.z *= mirror_vector.z

        print(f"search_vertex_position: {om2.MVector(source_vertex_position)}")

        result_indexes = []
        vertex_itr = om2.MItMeshVertex(dag_path)
        while not vertex_itr.isDone():
            dest_vertex_pos = vertex_itr.position(space)
            print(f"dest_vertex_pos: {om2.MVector(dest_vertex_pos)}")
            dest_vertex_index = vertex_itr.index()
            vertex_itr.next()

            if search_vertex_position == dest_vertex_pos:
                result_indexes.append(dest_vertex_index)
        print(result_indexes)

        return result_indexes

    def debug_print_selection_vertex_info(self):
        """
        デバック用。選択された頂点情報を出力する。
        """
        print("===== debug_print_selection_vertex_info =====")
        self._get_selection_vertices()
        for mesh_vertices_info in self.mesh_vertices_info_list:
            print(f"mesh: {mesh_vertices_info.dag_path.fullPathName()}")
            for face_index, vertex_infos in mesh_vertices_info.vertices.items():
                for vertex_info in vertex_infos:
                    print(
                        f"  vertex: {vertex_info.vertex_index},",
                        f"face: {face_index},",
                        f"weight: {math.floor(vertex_info.selection_weight*1000) / 1000}")
        print("=============================================")
