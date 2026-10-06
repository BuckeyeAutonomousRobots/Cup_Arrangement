"""ROS-independent bounded freshness recovery; all durations are wall seconds."""
class ObservationRecovery:
    def __init__(self):
        self.active = None
        self.events = []

    def update(self, wall, sim, image_stamp, joint_stamps, image_index, pending):
        stale = abs(sim-image_stamp) > .1 or any(abs(sim-t) > .1 for t in joint_stamps)
        if self.active is None and stale:
            if len(self.events) >= 3 or (self.events and wall-self.events[-1]['start_wall'] < 10):
                raise RuntimeError('Observation pauses too frequent')
            self.active = dict(start_wall=wall, start_sim=sim, image_index=image_index,
                               image_stamp=image_stamp, joint_stamps=list(joint_stamps),
                               image_age_sim_s=sim-image_stamp)
            self.events.append(self.active)
        if self.active is None:
            return 'normal'
        event = self.active
        event['duration_wall_s'] = wall-event['start_wall']
        if event['duration_wall_s'] >= .5:
            event['outcome'] = 'timeout'
            raise RuntimeError('Observation recovery exceeded 0.5 wall seconds')
        newer = image_index > event['image_index'] and image_stamp > event['image_stamp'] and all(
            t > old for t, old in zip(joint_stamps, event['joint_stamps']))
        if not stale and newer and not pending:
            event.update(outcome='fresh_observation', resume_sim=sim)
            self.active = None
            return 'resume'
        return 'wait'
