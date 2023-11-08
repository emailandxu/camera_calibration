import numpy as np

def projection(fov=90.0, aspect_ratio=1.0, near=0.1, far=1000.0):
    # Convert the fov from degrees to radians
    fov_rad = np.deg2rad(fov)
    
    # Calculate the scale using the fov
    scale = 1.0 / np.tan(fov_rad / 2.0)
    
    # Create the projection matrix
    return np.array([[scale / aspect_ratio, 0, 0, 0],
                     [0, scale, 0, 0],
                     [0, 0, -(far+near) / (far - near), -2 * far * near / (far - near)],
                     [0, 0, -1, 0]]).astype(np.float32)

                    
def translate(x, y, z):
    return np.array([[1, 0, 0, x], 
                     [0, 1, 0, y], 
                     [0, 0, 1, z], 
                     [0, 0, 0, 1]]).astype(np.float32)

def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1,  0, 0, 0], 
                     [0,  c, s, 0], 
                     [0, -s, c, 0], 
                     [0,  0, 0, 1]]).astype(np.float32)

def rotate_y(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[ c, 0, s, 0], 
                     [ 0, 1, 0, 0], 
                     [-s, 0, c, 0], 
                     [ 0, 0, 0, 1]]).astype(np.float32)

def rotate_z(a): # chat
    s, c = np.sin(a), np.cos(a)
    return np.array([[c, -s, 0, 0], 
                     [s,  c, 0, 0], 
                     [0,  0, 1, 0], 
                     [0,  0, 0, 1]]).astype(np.float32)

def scale(s):
    return np.array([[ s, 0, 0, 0], 
                     [ 0, s, 0, 0], 
                     [ 0, 0, s, 0], 
                     [ 0, 0, 0, 1]]).astype(np.float32)

def scale_xyz(x, y, z):
    return np.array([[ x, 0, 0, 0], 
                     [ 0, y, 0, 0], 
                     [ 0, 0, z, 0], 
                     [ 0, 0, 0, 1]]).astype(np.float32)

def lookAt(eye, at, up):
    a = eye - at
    b = up
    w = a / np.linalg.norm(a)
    u = np.cross(b, w)
    u = u / np.linalg.norm(u)
    v = np.cross(w, u)
    translate = np.array([[1, 0, 0, -eye[0]], 
                          [0, 1, 0, -eye[1]], 
                          [0, 0, 1, -eye[2]], 
                          [0, 0, 0, 1]]).astype(np.float32)
    rotate =  np.array([[u[0], u[1], u[2], 0], 
                        [v[0], v[1], v[2], 0], 
                        [w[0], w[1], w[2], 0], 
                        [0, 0, 0, 1]]).astype(np.float32)
    return np.matmul(rotate, translate)

def quat2mat(quat):
    """
    note that quat is assumed [x, y, z, w]
    """
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(quat).as_matrix()

def mat2quat(mat):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(mat[:3, :3]).as_quat()


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


def posemat(trans, quat, scale=None):
    rotmat = quat2mat(quat) # 3x3

    scale_mat = np.identity(4)
    
    if scale is not None:
        scale_mat[0, 0] = scale[0]
        scale_mat[1, 1] = scale[1]
        scale_mat[2, 2] = scale[2]

    return (np.array([
        [*rotmat[0], trans[0]],
        [*rotmat[1], trans[1]],
        [*rotmat[2], trans[2]],
        [0., 0., 0., 1.]
    ]) @ scale_mat).astype("f4")

def spherical(theta, phi, radius):
    # theta = np.radians(theta)
    # phi = np.radians(phi)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    x, y, z = y, z, x 
    return x, y, z

def checkerboard(width, repetitions) -> np.ndarray:
    tilesize = int(width//repetitions//2)
    check = np.kron([[1, 0] * repetitions, [0, 1] * repetitions] * repetitions, np.ones((tilesize, tilesize)))*0.33 + 0.33
    return np.stack((check, check, check), axis=-1)

if __name__ == "__main__":
    from scipy.spatial.transform import Rotation


    # print(Rotation.from_quat([0, 0, 0, 1]).as_matrix())
    # print(qvec2rotmat(np.array([1, 0, 0, 0])))

    print(qvec2rotmat(rotmat2qvec(rotate_z(np.pi/2)[:3,:3] @ np.identity(3))))
    print(Rotation.from_quat(Rotation.from_matrix(rotate_z(np.pi/2)[:3,:3] @ np.identity(3)).as_quat()).as_matrix())