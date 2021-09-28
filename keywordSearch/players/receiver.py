from datatype.datastore import Data
from learningMethods.charm_receiver import ReceiverCharmKeyword


class ReceiverKeyword(object):
    """docstring for ReceiverAll"""

    def __init__(self, receiverData, dataSource, datarecord, dbIndexSuffix=''):
        super(ReceiverKeyword, self).__init__()
        self.data = None
        self.receiverData = receiverData
        self.dataSource = dataSource
        self.datarecord = datarecord
        self.dbIndexSuffix = dbIndexSuffix

    def initializeRE(self):
        self.strategy = ReceiverCharmKeyword(self.data, self.receiverData, self.dataSource,
                                             self.datarecord, self.dbIndexSuffix)

    def setData(self, filename, filetype):
        self.data = Data(filename, filetype)

    def getHeader(self):
        return self.data.getHeader()

    def getTupleFromID(self, tupleID):
        return self.data.getListRow(tupleID)

    def getTuples(self, signals, numberToReturn=1):
        # Transform into format that the receiver expects
        signals = ([("('{}', '{}')".format(tup[0].keyword, 'UNUSED'), tup[1]) for tup in signals], [0])
        self.receivedSignal = signals
        self.returnedTuples = self.strategy.returnTuples(signals, numberToReturn)
        return self.returnedTuples
