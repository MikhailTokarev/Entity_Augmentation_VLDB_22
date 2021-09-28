from players.receiver import ReceiverKeyword
from players.sender import Sender
from players.sender_idf_baseline import SenderIDFBaseline
from players.oracle import Oracle
import shutil
import pickle
import os
import multiprocessing as mp

LOCAL_LEARNING = True # False = IDF Baseline

# Additional sender configuration
SELECTION_MODE = 'e-greedy' # 'e-greedy': Use epsilon greedy method when selecting keywords
							# 're': Use Roth and Erev method when selecting keywords

BORROW_TERMS = False

def save_obj(obj, name):
	with open(name + '.pkl', 'wb') as f:
		pickle.dump(obj, f)

def load_obj(name):
	with open('obj/' + name + '.pkl', 'rb') as f:
		return pickle.load(f)

def testWithFeatures(iterations, senderData, receiverData, oracleData, dataSource, numberOfTuplesToReturn,
					 numberOfSignalsToSend, resultsDirectory, reinforcementValue, dataRecord, averageRun):

	oracle = Oracle()
	oracle.initData(oracleData)

	# Set up the sender
	if LOCAL_LEARNING:
		sender = Sender()
	else:
		print('\t<USING IDFBaseline>')
		sender = SenderIDFBaseline()
	sender.setData(senderData, 'csv')
	sender.initialize(dataSource, dataRecord, SELECTION_MODE)

	# Set up the receiver
	receiver = ReceiverKeyword(receiverData, dataSource, dataRecord)
	receiver.setData(receiverData, 'csv')
	receiver.initializeRE()

	queryLog = list()

	# Statistics
	mostRecentRR = list()
	mostRecentPrecision = list()
	mostRecentRecall = list()

	receiver_tuples = dict()

	for iterationSoFar in range(1, iterations+1):
		'''
					Pick tuple in local, generate query based on tuple and retrieve results from external
		'''
		tuplesToSend = sender.pickTupleToJoin()
		tuplesReturnedInOrder = list()
		queryToSend = sender.pickSignals(tuplesToSend, numberOfSignalsToSend)
		tuplesReturnedInOrder.append(receiver.getTuples(queryToSend, numberOfTuplesToReturn))

		'''
					Check external results for matches and update models
		'''
		foundIt = False
		matchesFound = 0
		reciprocalRank = 0

		if LOCAL_LEARNING and BORROW_TERMS and iterationSoFar == 1000:
			receiver_data = list(receiver_tuples.values())
			sender.processExternalData(receiver_data)

		for idSent in tuplesToSend:
			for tuplesReturnedTotal in tuplesReturnedInOrder:
				position = 1
				for idReceived in tuplesReturnedTotal:
					if idReceived not in receiver_tuples:
						receiver_tuples[idReceived] = receiver.getTupleFromID(idReceived)

					if oracle.getTruth(idSent, idReceived):

						matchesFound += 1
						if not foundIt:
							reciprocalRank = 1 / position
						foundIt = True

						'''
											Update Local
						'''
						terms_sent = [pair[0] for pair in queryToSend]

						if LOCAL_LEARNING and BORROW_TERMS:
							sender.addTerms(idSent, receiver.getTupleFromID(idReceived))

						if SELECTION_MODE == 'e-greedy':
							sender.update_model_shared(terms_sent, idSent,
														   receiver.strategy.numOfFeatures[idReceived],
														   reciprocalRank)
						else:
							terms_in_matching_tuple = receiver.strategy.numOfFeatures[idReceived]
							receivedFeatures = [f.split(',')[0][1:].replace('\'', '') for f in
												terms_in_matching_tuple]

							queryExternalKeywordOverlap = []

							# Reinforce the union of keywords sent and keywords in external match
							for signalTup in queryToSend:
								if signalTup[0].keyword in receivedFeatures:
									queryExternalKeywordOverlap.append(signalTup[0].keyword)

							sender.reinforceSelectionStrategy(idSent, queryExternalKeywordOverlap, reinforcementValue)

					position += 1

			if not foundIt:

				# Assumes the reward for the keywords was 0 due to no match being observed
				# Only update model if it has been updated from finding a match
				if sender.modelIsFitted():
					terms_sent = [pair[0] for pair in queryToSend]
					sender.update_model_shared(terms_sent, idSent, None, reciprocalRank)

		'''
					Update statistics
		'''
		precision = matchesFound / max(len(tuplesReturnedInOrder[0]), 1)
		recall = matchesFound / max(oracle.getTotalTrue(tuplesToSend[0]), 1)

		mostRecentRR.append(reciprocalRank)
		mostRecentRecall.append(recall)
		mostRecentPrecision.append(precision)

		queryLog.append((tuplesToSend, queryToSend, reciprocalRank))

		'''
					Print/Save statistics
		'''
		if iterationSoFar % 500 == 0:
			print(dataSource + '_' + str(iterationSoFar) + '_returned=' + str(numberOfTuplesToReturn) + '_sent=' + str(
				numberOfSignalsToSend) + '_AverageRun=' + str(averageRun))
			print('MRR: ' + str(sum(mostRecentRR) / iterationSoFar))
			print('Average MRR (last 500): ' + str(sum(mostRecentRR[len(mostRecentRR) - 500:]) / 500))
			print('Average Recall (last 500): ' + str(sum(mostRecentRecall[len(mostRecentRecall) - 500:]) / 500))

			print()

		if iterationSoFar % 1000 == 0:
			zip_dir = resultsDirectory + '/' + str(iterationSoFar) + '_AverageRun=' + str(averageRun)
			os.makedirs(zip_dir)

			filePre = zip_dir
			filePost = dataSource + '_' + str(iterationSoFar) + 'reinforcement=' + str(reinforcementValue) \
					   + '_AverageRun=' + str(averageRun) + '_returned=' + str(numberOfTuplesToReturn) + '_sent=' + str(
				numberOfSignalsToSend)

			print('saving')

			save_obj(sender, '{}/sender_{}'.format(filePre, filePost))
			save_obj(receiver, '{}/receiver_{}'.format(filePre, filePost))

			save_obj(mostRecentRR, '{}/mostRecentMRR_{}'.format(filePre, filePost))
			save_obj(mostRecentRecall, '{}/mostRecentRecall_{}'.format(filePre, filePost))
			save_obj(mostRecentPrecision, '{}/mostRecentPrecision_{}'.format(filePre, filePost))

			save_obj(queryLog, '{}/queryLog_{}'.format(filePre, filePost))

			shutil.make_archive(zip_dir, 'zip', zip_dir)
			shutil.rmtree(zip_dir)

if __name__ == '__main__':
	iterations = 2000  # iterations to run (i.e., amount of times we try to match tuples)
	numberOfTuplesToReturn = 20  # top-k tuples returned from externally
	numberOfSignalsToSend = 4  # keywords to send (length of query)
	reinforcementValue = 100
	dataset = 'amazon'
	averageRuns = 1  # Spawns <averageRuns> processes

	# Filepath for where experiment results will be saved
	resultsPre = '/data/experiments/' + dataset + '/'

	# Sender config:
	if not LOCAL_LEARNING:
		resultsPre += 'idf'
	else:
		resultsPre += SELECTION_MODE

	# Any additional modifiers
	resultsPre += '-borrow_terms={}/'.format(BORROW_TERMS)

	# Directory for experiments in filepath above (i.e., <resultsPre>/<resultsFilename>)
	resultsFilename = 'keys_' + str(numberOfSignalsToSend)

	# Any additional modifiers
	resultsFilename += '/'
	resultsDirectory = resultsPre + resultsFilename

	# Suffix for pre-processed data.
	#	NOTE: also change RECORD_DIR_PATH in datatype/wooshengine.py, learningMethods/charm_receiver.py, and learningMethods/charm_sender.py
	#		pre-processed data saved in <RECORD_DIR_PATH>/<DATASET_NAME><dataRecord>
	dataRecord = ''

	if not os.path.exists(resultsDirectory):
		print(resultsDirectory + " does not exist. Create it (y/n)?")
		print('Averaged runs: {}'.format(averageRuns))
		if input() != 'y':
			exit()
		else:
			print('creating: ' + resultsDirectory)
			os.makedirs(resultsDirectory)
	else:
		print(resultsDirectory + " exists. Continue (y/n)?")
		print('Averaged runs: {}'.format(averageRuns))
		if input() != 'y':
			exit()

	senderData = 'datasets/' + dataset + '/source.csv'
	receiverData = 'datasets/' + dataset + '/target.csv'
	oracleData = 'datasets/' + dataset + '/ground_truth.csv'

	process_list = []
	for i in range(averageRuns):
		p = mp.Process(target=testWithFeatures, args=(iterations, senderData, receiverData, oracleData, dataset,
													  numberOfTuplesToReturn, numberOfSignalsToSend, resultsDirectory,
													  reinforcementValue,
													  dataRecord, i + 1))
		p.start()
		process_list.append(p)

	for p in process_list:
		p.join()

	print('Complete.')
