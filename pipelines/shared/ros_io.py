#!/usr/bin/env python3
"""ros_io.py – Shared ROS 2 message deserialization helpers."""
import bisect, math, sys
from io import BytesIO
from pathlib import Path
import numpy as np
import open3d as o3d
from PIL import Image
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation as R

TYPESTORE = get_typestore(Stores.ROS2_HUMBLE)
_POINTFIELD_TO_DTYPE = {7: np.float32, 8: np.float64}
_GPS_NO_FIX = -1

try:
    from tf_transformations import quaternion_matrix as _qm
    quaternion_matrix = _qm
except ImportError:
    def quaternion_matrix(q):
        x,y,z,w = q; n=(x*x+y*y+z*z+w*w)**0.5; x,y,z,w=x/n,y/n,z/n,w/n
        m=np.zeros((4,4)); m[3,3]=1.0
        m[0,0]=1-2*(y*y+z*z); m[0,1]=2*(x*y-z*w); m[0,2]=2*(x*z+y*w)
        m[1,0]=2*(x*y+z*w);   m[1,1]=1-2*(x*x+z*z); m[1,2]=2*(y*z-x*w)
        m[2,0]=2*(x*z-y*w);   m[2,1]=2*(y*z+x*w);   m[2,2]=1-2*(x*x+y*y)
        return m

def convert_ros_pc2_to_o3d(msg):
    try:
        fields={f.name:(int(f.offset),int(f.datatype)) for f in msg.fields}
        if not all(k in fields for k in ("x","y","z")): return None
        xo,xd=fields["x"]; yo,yd=fields["y"]; zo,zd=fields["z"]
        if xd not in _POINTFIELD_TO_DTYPE or xd!=yd or yd!=zd: return None
        dt=_POINTFIELD_TO_DTYPE[xd]; n=int(msg.width)*int(msg.height)
        dtype=np.dtype({"names":["x","y","z"],"formats":[dt,dt,dt],
                        "offsets":[xo,yo,zo],"itemsize":int(msg.point_step)})
        arr=np.frombuffer(msg.data,dtype=dtype,count=n)
        pts=np.empty((n,3),np.float64); pts[:,0]=arr["x"]; pts[:,1]=arr["y"]; pts[:,2]=arr["z"]
        pts=pts[np.isfinite(pts).all(1)]
        if len(pts)<10: return None
        pcd=o3d.geometry.PointCloud(); pcd.points=o3d.utility.Vector3dVector(pts); return pcd
    except Exception: return None

def pointcloud2_to_numpy(msg):
    try:
        from robotdatapy.pointcloud.pointcloud_conversions import pointcloud2_to_xyz_array
        return pointcloud2_to_xyz_array(msg)
    except Exception: pass
    try:
        n=msg.height*msg.width
        if n==0: return np.array([])
        xo=yo=zo=None
        for f in msg.fields:
            if f.name=="x": xo=f.offset
            elif f.name=="y": yo=f.offset
            elif f.name=="z": zo=f.offset
        if None in (xo,yo,zo): return np.array([])
        data=bytes(msg.data); step=msg.point_step; pts=[]
        for i in range(n):
            o=i*step
            x=np.frombuffer(data[o+xo:o+xo+4],np.float32)[0]
            y=np.frombuffer(data[o+yo:o+yo+4],np.float32)[0]
            z=np.frombuffer(data[o+zo:o+zo+4],np.float32)[0]
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z): pts.append((x,y,z))
        return np.array(pts,np.float32) if pts else np.array([])
    except Exception: return np.array([])

def get_odom_transform(msg):
    try:
        p=msg.pose.pose.position; q=msg.pose.pose.orientation
        T=np.eye(4,dtype=np.float64)
        T[:3,:3]=R.from_quat([q.x,q.y,q.z,q.w]).as_matrix()
        T[:3,3]=[p.x,p.y,p.z]; return T
    except Exception: return None

def get_odom_transform_matrix(msg):
    try:
        p=msg.pose.pose.position; o=msg.pose.pose.orientation
        m=quaternion_matrix([o.x,o.y,o.z,o.w])
        m[0,3]=p.x; m[1,3]=p.y; m[2,3]=p.z; return m
    except Exception: return None

def get_closest_timestamp(ts, sorted_keys):
    if not sorted_keys: return None
    i=bisect.bisect_left(sorted_keys,ts)
    if i==0: return sorted_keys[0]
    if i==len(sorted_keys): return sorted_keys[-1]
    b,a=sorted_keys[i-1],sorted_keys[i]
    return b if (ts-b)<=(a-ts) else a

def convert_ros_image(msg):
    try:
        if hasattr(msg,"format"): return Image.open(BytesIO(bytes(msg.data))).convert("RGB")
        enc=getattr(msg,"encoding","bgr8"); data=bytes(msg.data); h,w=int(msg.height),int(msg.width)
        if enc in("bgr8","rgb8","8UC3"):
            arr=np.frombuffer(data,np.uint8).reshape(h,w,3)
            if enc=="bgr8": arr=arr[:,:,::-1]
            return Image.fromarray(arr,"RGB")
        if enc in("mono8","8UC1"):
            return Image.fromarray(np.frombuffer(data,np.uint8).reshape(h,w),"L").convert("RGB")
        return Image.open(BytesIO(data)).convert("RGB")
    except Exception: return None

def intrinsics_from_camera_info(msg):
    try:
        K=list(msg.k)
        return float(K[0]),float(K[4]),float(K[2]),float(K[5]),int(msg.width),int(msg.height)
    except Exception: return None

def parse_gps_fixes(bag_path, gps_topic):
    bag_path=Path(bag_path); fixes=[]
    with AnyReader([bag_path],default_typestore=TYPESTORE) as reader:
        conns=[c for c in reader.connections if c.topic==gps_topic]
        if not conns: raise RuntimeError(f"GPS topic '{gps_topic}' not found in bag.")
        for conn,_ts,raw in reader.messages(connections=conns):
            try:
                msg=reader.deserialize(raw,conn.msgtype)
                if int(msg.status.status)==_GPS_NO_FIX: continue
                lat,lon,alt=float(msg.latitude),float(msg.longitude),float(msg.altitude)
                if all(math.isfinite(v) for v in (lat,lon,alt)): fixes.append((lat,lon,alt))
            except Exception: continue
    if not fixes: raise RuntimeError(f"No valid GPS fixes found on topic '{gps_topic}'.")
    n=len(fixes)
    lat0=sum(f[0] for f in fixes)/n; lon0=sum(f[1] for f in fixes)/n; alt0=sum(f[2] for f in fixes)/n
    print(f"GPS: {n} fix(es) averaged -> lat={lat0:.7f} lon={lon0:.7f} alt={alt0:.3f} m")
    return lat0,lon0,alt0
