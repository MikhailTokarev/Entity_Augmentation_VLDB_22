import multiprocessing as mp
import pickle

from additionalScripts.utils.calculateOptimalQueries import calculateOptimalQueries
from players.oracle import Oracle
from players.receiver import ReceiverKeyword
from players.sender import Sender

if __name__ == '__main__':
    DATASET = 'amazon'

    # Set up the sender
    sender = Sender()
    senderData = '../datasets/{}/source.csv'.format(DATASET)
    sender.setData(senderData, 'csv')
    sender.initialize(DATASET, '', 'e-greedy')
    del sender.data.table

    oracleData = '../datasets/{}/ground_truth.csv'.format(DATASET)
    oracle = Oracle()
    oracle.initData(oracleData)

    # load sample list (pickled list of IDs)
    sample_list = pickle.load(open('{}-samplelist.pkl'.format(DATASET), 'rb'))
    # sample_list = [tupID for tupID in sender.strategy.signalIndex.keys()]

    K = 20
    MIN = 1
    MAX = 3
    PROCESSES = 1
    SEGMENT_OFFSET = 0
    SEGMENT_LENGTH = len(sample_list)
    BREAK_PART = [1, 1]  # [x, y] x of y parts
    PART_LENGTH = SEGMENT_LENGTH / BREAK_PART[1]
    PART_START = int(PART_LENGTH*(BREAK_PART[0]-1)) + SEGMENT_OFFSET
    PART_END = int(PART_LENGTH*(BREAK_PART[0])) + SEGMENT_OFFSET

    print('Here: {}'.format(SEGMENT_LENGTH))
    print('{} - {}'.format(PART_START, PART_END))

    thread_list = []

    print('Starting {} processes... on {} with queries of size {}-{}'.format(PROCESSES, DATASET, MIN, MAX))

    keys_per_thread = int((PART_END - PART_START) / PROCESSES)
    for i in range(PROCESSES):

        # Set up the receiver
        receiverData = '../datasets/{}/target.csv'.format(DATASET)
        receiver = ReceiverKeyword(receiverData, DATASET, '')
        receiver.setData(receiverData, 'csv')
        receiver.initializeRE()
        del receiver.data.table

        offset = PART_START + (i * keys_per_thread)
        if i == PROCESSES - 1:
            t = mp.Process(target=calculateOptimalQueries,
                           args=(DATASET, sender, receiver, oracle, offset, PART_END, MIN, MAX, K, sample_list))
        else:
            print()
            t = mp.Process(target=calculateOptimalQueries,
                           args=(DATASET, sender, receiver, oracle, offset, offset + keys_per_thread, MIN, MAX, K, sample_list))
        t.start()
        thread_list.append(t)

    for t in thread_list:
        t.join()
