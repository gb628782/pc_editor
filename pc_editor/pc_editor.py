import bresenham
from math import sin, cos, pi,tan, atan2,log
import math
from itertools import groupby
from operator import itemgetter
import rclpy
from rclpy.node import Node
import numpy as np
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import MapMetaData
from geometry_msgs.msg import Pose
import math
from sensor_msgs.msg import PointCloud2, PointField
import struct
import ctypes
import time

type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')), (PointField.INT16, np.dtype('int16')),
                  (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')), (PointField.UINT32, np.dtype('uint32')),
                  (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

_DATATYPES = {}
_DATATYPES[PointField.INT8]    = ('b', 1)
_DATATYPES[PointField.UINT8]   = ('B', 1)
_DATATYPES[PointField.INT16]   = ('h', 2)
_DATATYPES[PointField.UINT16]  = ('H', 2)
_DATATYPES[PointField.INT32]   = ('i', 4)
_DATATYPES[PointField.UINT32]  = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)

def _get_struct_fmt(is_bigendian, fields, field_names=None):
     fmt = '>' if is_bigendian else '<'
 
     offset = 0
     for field in (f for f in sorted(fields, key=lambda f: f.offset) if field_names is None or f.name in field_names):
         if offset < field.offset:
             fmt += 'x' * (field.offset - offset)
             offset = field.offset
         if field.datatype not in _DATATYPES:
             print('Skipping unknown PointField datatype [%d]' % field.datatype, file=sys.stderr)
         else:
             datatype_fmt, datatype_length = _DATATYPES[field.datatype]
             fmt    += field.count * datatype_fmt
             offset += field.count * datatype_length
 
     return fmt

def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
     """
     Read points from a L{sensor_msgs.PointCloud2} message.
 
00064     @param cloud: The point cloud to read from.
00065     @type  cloud: L{sensor_msgs.PointCloud2}
00066     @param field_names: The names of fields to read. If None, read all fields. [default: None]
00067     @type  field_names: iterable
00068     @param skip_nans: If True, then don't return any point with a NaN value.
00069     @type  skip_nans: bool [default: False]
00070     @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
00071     @type  uvs: iterable
00072     @return: Generator which yields a list of values for each point.
00073     @rtype:  generator
     """
     fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
     width, height, point_step, row_step, data, isnan = cloud.width, cloud.height, cloud.point_step, cloud.row_step, cloud.data, math.isnan
     unpack_from = struct.Struct(fmt).unpack_from
     count = 0
     if skip_nans:
         if uvs:
             for u, v in uvs:
                 p = unpack_from(data, (row_step * v) + (point_step * u))
                 has_nan = False
                 for pv in p:
                     if isnan(pv):
                         has_nan = True
                         break
                 if not has_nan:
                     count += 1
                     if count == 200:
                         count = 0
                         yield p
         else:
             for v in range(0, height, 10):
                 offset = row_step * v
                 for u in range(0, width, 20):
                     p = unpack_from(data, offset)
                     has_nan = False
                     for pv in p:
                         if isnan(pv):
                             has_nan = True
                             break
                     if not has_nan:
                         yield p
                     offset += point_step*20
     else:
         if uvs:
             for u, v in uvs:
                 count += 1
                 if count == 200:
                     count = 0
                     yield unpack_from(data, (row_step * v) + (point_step * u))
         else:
             for v in range(height):
                 offset = row_step * v
                 for u in range(width):
                     count += 1
                     if count == 200:
                         count = 0
                         yield unpack_from(data, offset)
                     offset += point_step


def dtype_to_fields(dtype):
     '''Convert a numpy record datatype into a list of PointFields.
     '''
     fields = []
     for field_name in dtype.names:
         np_field_type, field_offset = dtype.fields[field_name]
         pf = PointField()
         pf.name = field_name
         if np_field_type.subdtype:
             item_dtype, shape = np_field_type.subdtype
             pf.count = np.prod(shape)
             np_field_type = item_dtype
         else:
             pf.count = 1
 
         pf.datatype = nptype_to_pftype[np_field_type]
         pf.offset = field_offset
         fields.append(pf)
     return fields

def array_to_pointcloud2(cloud_arr, stamp=None, frame_id=None):
    '''Converts a numpy record array to a sensor_msgs.msg.PointCloud2.
    '''
    # make it 2d (even if height will be 1)
    cloud_arr = np.atleast_2d(cloud_arr)

    cloud_msg = PointCloud2()

    if stamp is not None:
        cloud_msg.header.stamp = stamp
    if frame_id is not None:
        cloud_msg.header.frame_id = frame_id
    cloud_msg.height = cloud_arr.shape[0]
    cloud_msg.width = cloud_arr.shape[1]
    cloud_msg.fields = dtype_to_fields(cloud_arr.dtype)
    cloud_msg.is_bigendian = False # assumption
    cloud_msg.point_step = cloud_arr.dtype.itemsize
    cloud_msg.row_step = cloud_msg.point_step*cloud_arr.shape[1]
    cloud_msg.is_dense = all([np.isfinite(cloud_arr[fname]).all() for fname in cloud_arr.dtype.names])
    cloud_msg.data = cloud_arr.tostring()
    return cloud_msg 

class PCPubSub(Node):
    def __init__(self):
        super().__init__('pc_editor')
        self.publisher_ = self.create_publisher(PointCloud2, '/modpointcloud', 1)
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/aligned_depth_to_color/color/points',
            self.sub_callback,
            5)
        self.subscription  # prevent unused variable warning


    def sub_callback(self, pcl):
        lst = []
        cloud_it = list(read_points(pcl, field_names = ('x', 'y', 'z'), skip_nans = True))

        lst = [p for p in cloud_it if -0.1 < p[1] < 0.1]
        recarr = np.array(lst, dtype=[('x', float), ('y', float), ('z', float)])
        recarr = recarr.view(np.recarray)
        new_pcl = array_to_pointcloud2(recarr)
        self.publisher_.publish(new_pcl)


def main():
    rclpy.init()
    pc_editor_node = PCPubSub()
    rclpy.spin(pc_editor_node)
    grid_map_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
