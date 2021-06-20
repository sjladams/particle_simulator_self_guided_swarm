import struct
import time
import math
import numpy as np
from scipy.spatial.transform import Rotation as R

WHEEL_DIAMETER = 0.009
STD_SPEED_TRANS = 28      # 13.5
# STD_SPEED_TRANS = 6.28 * (WHEEL_DIAMETER*np.pi)     # 13.5
STD_SPEED_TRANS_RAD = int(1 / (WHEEL_DIAMETER*np.pi))     # 0.1
STD_SPEED_ROT = 260          # 25
STD_SPEED_ROT_RAD = int(1 / (WHEEL_DIAMETER*np.pi))       # 0.1

braitenberg_coefficients = [[10, 8], [7, -1.5], [5, -1], [-1, -1], [-1, -1], [-1, -1], [-1, 5], [-1.5, 7]]
RANGE = (127 / 2)
WHEEL_DISTANCE = 0.0408

class RobotController():
    def __init__(self, tag, robot_name, supervisor,timestep):
        self.tag = tag

        self.node = supervisor.getFromDef(robot_name)
        self.emitter = self.node.getField('emitter_channel')
        self.receiver = self.node.getField('receiver_channel')

        self.trans_field = self.node.getField('translation')
        self.rot_field = self.node.getField('rotation')

        self.emitter.setSFInt32(tag)
        self.receiver.setSFInt32(tag)

        self.super_receiver = supervisor.getDevice('receiver-' + str(tag))
        self.super_receiver.enable(timestep)
        self.super_receiver.setChannel(tag)

        self.txData = [0 for _ in range(0, 10)]
        self.motionState = 0
        self.controlSteps = 0 #turn off for 100 steps
        self.colorState = 0
        self.timeout = 0

        self.init_location_orientation()

    def checkMotionState(self):
        # self.waiting_time = int(self.controlSteps / (32 * 1e-6))
        # self.waiting_time = int((self.trans_controlSteps + self.orien_controlSteps / 85) / (32 * 1e-6))
        self.waiting_time = int(self.trans_controlSteps * 1.5 / (32 * 1e-6))

        self.txData[3] = int(self.orien_controlSteps)
        self.txData[4] = self.orien_motionState
        self.txData[5] = self.trans_motionState

        steps2com = str(int((self.trans_controlSteps) / (32 * 1e-6)))
        list2com = list(reversed([steps2com[i:(i+2)] for i in range(0,len(steps2com),2)]))
        self.txData[6:] = [0, 0, 0, 0]
        for count, elem in enumerate(list2com):
            self.txData[-count-1] = int(elem)

    def checkColorState(self, timestep):
        if self.colorState == 0:
            self.txData[0] = 100
            self.txData[1] = 0
            self.txData[2] = 0
            self.timeout += 1
            if self.timeout >= 1000 / timestep:
                colorState = 1
        elif self.colorState == 1:
            self.txData[0] = 0
            self.txData[1] = 0
            self.txData[2] = 100
            self.timeout += 1
            if self.timeout >= 1000 / timestep:
                colorState = 2
        elif self.colorState == 2:
            self.txData[0] = 0
            self.txData[1] = 100
            self.txData[2] = 0
            self.timeout += 1
            if self.timeout >= 1000 / timestep:
                colorState = 3
        elif self.colorState == 3:
            self.txData[0] = 100
            self.txData[1] = 100
            self.txData[2] = 100
            self.timeout += 1
            if self.timeout >= 1000 / timestep:
                colorState = 0

    def SendMsg(self, emitter):
        emitter.setChannel(self.tag)
        message = struct.pack('b' * 10, self.txData[0],self.txData[1],self.txData[2],self.txData[3],self.txData[4],
                              self.txData[5],self.txData[6],self.txData[7],self.txData[8],self.txData[9])
        emitter.send(message)

    def prepare_step(self, emitter, pol_move = np.array([0., 0])):
        self.init_location_orientation()
        self.update_location_orientation()
        self.set_targ(pol=pol_move)

    def update_step(self, emitter):
        self.update_location_orientation()
        self.checkMotionState()
        self.SendMsg(emitter)

    def CheckReceiver(self):
        if self.super_receiver.getQueueLength() > 0:
            msg = struct.unpack('b'*81,self.super_receiver.getData())
            self.super_receiver.nextPacket()

    def init_location_orientation(self):
        self.init_location = np.array(self.trans_field.getSFVec3f())

        rot = self.rot_field.getSFRotation()
        self.init_orien = rot[3] % (2*np.pi)
        # self.init_orien = self.yaw_from_axisAngle(rot)

    def update_location_orientation(self):
        self.cur_location = np.array(self.trans_field.getSFVec3f())
        delta_location = self.cur_location - self.init_location
        self.travl_dist = math.sqrt(delta_location[0] ** 2 + delta_location[2] ** 2)

        rot = self.rot_field.getSFRotation()

        # self.cur_orien = self.yaw_from_axisAngle(rot)
        self.cur_orien = - rot[3] % (2*np.pi)
        # delta_orien = self.cur_orien - self.init_orien

        print("dist=%g, x=%g, y=%g" % (self.travl_dist, self.cur_location[0], self.cur_location[2]))
        print("current yaw=%g" % (self.cur_orien))

    @staticmethod
    def yaw_from_axisAngle(rot):
        x = rot[0] * np.sin(rot[3]/2)
        y = rot[1] * np.sin(rot[3]/2)
        z = rot[2] * np.sin(rot[3]/2)
        w = np.cos((rot[3] %(2*np.pi))/2)
        # t3 = 2.0 * (w*z + x*y)
        # t4 = 1.0 - 2.0*(y*y+z*z)
        # yaw_z = math.atan2(t3, t4)
        r = R.from_quat([x, y, z, w])
        euler_angles = r.as_euler('zyx')
        return euler_angles[1]

    def set_targ(self, pol=np.array([0,0])):
        phi = pol[1] - self.cur_orien
        if phi < 0:
            phi += (2*np.pi)
        phi %= (2*np.pi)

        if phi <= 0.5*np.pi:
            # turn
            self.orien_motionState = 1
            # self.orien_controlSteps = (WHEEL_DISTANCE*phi) / STD_SPEED_ROT
            self.orien_controlSteps = phi * (180 / np.pi)
            # self.orien_targ = phi
            # trans
            # self.trans_motionState = 0
            self.trans_motionState = 1
            self.trans_controlSteps = pol[0] / STD_SPEED_TRANS
            # self.travl_targ = pol[0]
        elif phi <= np.pi:
            # turn
            # self.orien_motionState = 2
            self.orien_motionState = 0
            # self.orien_controlSteps = (WHEEL_DISTANCE)*(np.pi - phi) / STD_SPEED_ROT
            self.orien_controlSteps = (np.pi - phi) * (180 / np.pi)
            # self.orien_targ = (np.pi - phi)
            # trans
            # self.trans_motionState = 3
            self.trans_motionState = 0
            self.trans_controlSteps = pol[0] / STD_SPEED_TRANS
            # self.travl_targ = pol[0]
        elif phi <= 1.5*np.pi:
            # turn
            self.orien_motionState = 1
            # self.orien_controlSteps = (WHEEL_DISTANCE)*(phi - np.pi) / STD_SPEED_ROT
            self.orien_controlSteps = (phi - np.pi) * (180 / np.pi)
            # self.orien_targ = (phi - np.pi)
            # trans
            # self.trans_motionState = 3
            self.trans_motionState = 0
            self.trans_controlSteps = pol[0] / STD_SPEED_TRANS
            # self.travl_targ = pol[0]
        else:
            # turn
            # self.orien_motionState = 2
            self.orien_motionState = 0
            # self.orien_controlSteps = (WHEEL_DISTANCE) * (2*np.pi - phi) / STD_SPEED_ROT
            self.orien_controlSteps = (2*np.pi - phi) * (180 / np.pi)
            # trans
            # self.trans_motionState = 0
            self.trans_motionState = 1
            self.trans_controlSteps = pol[0] / STD_SPEED_TRANS
            # self.travl_targ = pol[0]

