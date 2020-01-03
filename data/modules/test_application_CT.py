import simpy
from random import randint
env = simpy.Environment()

CT_NUMBER = 4
NORMAL_PATIENT_NUMBER = 100
STROKE_PATIENT_NUMBER = 75

StrokeWaitTimeList = []

CT = simpy.PriorityResource(env, capacity=CT_NUMBER)

def patient(name, category, env, CT, ETA, prio, scan_time):
    yield env.timeout(ETA)

    with CT.request(priority=prio) as req:
        start = env.now

        print('%s requests CT at %d with priority=%s' % (name, env.now, prio))
        yield req
        print('%s entered CT at %d' % (name, env.now))
        yield env.timeout(scan_time)
        print('%s finished scan at %d minutes' % (name, env.now))
        localwait=float(env.now-start)
        if category is 'stroke':
            print('that scan took %d minutes' % (localwait))
            StrokeWaitTimeList.append(localwait)
            print(StrokeWaitTimeList)
            AvgStrokeWaitTime = sum(StrokeWaitTimeList)/len(StrokeWaitTimeList)
            print(AvgStrokeWaitTime)
            print('AVG stroke wait time is %.2f minutes' % AvgStrokeWaitTime)



for i in range(NORMAL_PATIENT_NUMBER):
    env.process(patient('np %d' % i, 'normal', env, CT, randint(0,NORMAL_PATIENT_NUMBER/10), 1, randint(1,5)))
    #env.process(patient('np %d' % i, 'normal', env, CT, NORMAL_PATIENT_NUMBER, 1, randint(1,5)))

for i in range(STROKE_PATIENT_NUMBER):
    env.process(patient('STROKEp %d' % i, 'stroke', env, CT, randint(0,NORMAL_PATIENT_NUMBER*2), 0, 5))


env.run()
