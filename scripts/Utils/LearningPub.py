import rospy

from Utils.Utils import p
from mpdm.msg import Ped, Peds, Propagation, Learning

class LearningPub:
    def __init__(self, topic_name="mpdm/debug"):
        self.pub_learning = rospy.Publisher(topic_name, Learning, queue_size=0)

    def publish(self, states, goals, costs, covariances, policys, propagation_times, learning_time, frame = "map"):
        msg = Learning()
        msg.learning_time.data = learning_time
        for epoch in range(len(states)):
            prop = Propagation()
            prop.cost.data = costs[epoch]
            prop.policy.data = policys[epoch]
            prop.propagation_time.data = propagation_times[epoch]
            for step in range(len(states[epoch])):
                peds = Peds()
                for pd in range(len(states[epoch][step])):
                    ped = Ped()
                    ped.header.frame_id = frame
                    ped.id.data = str(pd) # TODO: think about how to return id from mpdm
                    # ebuchaya konvertaciya tipov
                    ped.position = p(float(states[epoch][step][pd][0]), float(states[epoch][step][pd][1]), float(states[epoch][step][pd][2]))
                    ped.velocity = p(float(states[epoch][step][pd][3]), float(states[epoch][step][pd][4]), float(states[epoch][step][pd][5]))
                    ped.goal = p(float(goals[pd][0]), float(goals[pd][1]), float(goals[pd][2]))
                    ped.cov_pose = p(covariances[epoch][step][pd][0], covariances[epoch][step][pd][1])
                    peds.peds.append(ped)
                prop.steps.append(peds)
            msg.epochs.append(prop)

        self.pub_learning.publish(msg)
