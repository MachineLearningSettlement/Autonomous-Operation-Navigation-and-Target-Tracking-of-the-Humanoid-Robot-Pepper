import time
import math
from naoqi import ALProxy

# ==========================================
# Robot Configuration
# ==========================================

ip = "10.12.20.165"
port = 9559

motion = ALProxy("ALMotion", ip, port)
posture = ALProxy("ALRobotPosture", ip, port)

# ==========================================
# Get target object coordinates
# ==========================================

def getTargetObject():
    """
    Retrieves the Cartesian coordinates (X, Y, Z)
    of the first detected object from
    objects_coordinates.txt.

    File format:
    object_name, confidence, center_x, center_y, depth_z, X, Y, Z
    """

    with open("objects_coordinates.txt", "r") as file:

        line = file.readline().strip()

        data = [item.strip() for item in line.split(",")]

        x = float(data[5])   # X
        y = float(data[6])   # Y
        z = float(data[7])   # Z

    return x, y, z


# ==========================================
# Simplified Inverse Kinematics
# ==========================================

def inverseKinematics(x, y, z):

    L1 = 0.18
    L2 = 0.22

    r = math.sqrt(x**2 + y**2)
    d = math.sqrt(r**2 + z**2)
    d = min(d, L1 + L2 - 1e-6)

    shoulderRoll = math.atan2(y, x)

    c = (L1**2 + L2**2 - d**2) / (2 * L1 * L2)
    c = max(-1.0, min(1.0, c))
    elbowRoll = math.pi - math.acos(c)

    alpha = math.atan2(z, r)
    beta = math.acos((L1**2 + d**2 - L2**2) / (2 * L1 * d))
    shoulderPitch = alpha + beta

    elbowYaw = -shoulderRoll
    wristYaw = 0.0

    return [
        shoulderPitch,
        shoulderRoll,
        elbowYaw,
        elbowRoll,
        wristYaw
    ]


# ==========================================
# Main Program
# ==========================================

# Enable motors
motion.setStiffnesses("Body", 1.0)

# Stand up
posture.goToPosture("StandInit", 0.5)

# Read target object coordinates
x, y, z = getTargetObject()

# Compute joint angles
angles = inverseKinematics(x, y, z)

jointNames = [
    "RShoulderPitch",
    "RShoulderRoll",
    "RElbowYaw",
    "RElbowRoll",
    "RWristYaw"
]

motion.setAngles(jointNames, angles, 0.2)

# Close hand
motion.setStiffnesses("RHand", 1.0)
motion.setAngles("RHand", 0.0, 0.2)

time.sleep(1)

# Keep arm fixed during navigation
motion.setMoveArmsEnabled(False, False)

# Navigate while holding the object
motion.moveTo(1.0, 0.0, 0.0)

# Restore arm motion
motion.setMoveArmsEnabled(True, True)
