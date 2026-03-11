import os
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = '../DATASET/NETWORK/norway_raw/'
FILE_BW = '../DATASET/NETWORK/norway_raw/bus.ljansbakken-oslo.report.2010-09-28_1407CEST.log'

OUTPUT_PATH = '../DATASET/NETWORK/norway_mahimahi/'
FILE_MAHIMAHI = '../DATASET/NETWORK/norway_mahimahi/norway_bus_1'


TIME_INTERVAL = 5.0
PACKET_SIZE = 1500.0  # bytes
MBITS_IN_BITS = 1000000.0
N = 100

CHUNK_DURATION = 320.0  # duration in seconds
CHUNK_JUMP = 60.0  # shift in seconds

BYTES_PER_PKT = 1500.0
MILLISEC_IN_SEC = 1000.0
BITS_IN_BYTE = 8.0



def convert_mahimahi_norway():
	files = os.listdir(DATA_PATH)

	for f in files:
		file_path = DATA_PATH +  f
		output_path = OUTPUT_PATH + f

		print(file_path)

		with open(file_path, 'r') as f, open(output_path, 'w') as mf:
			time_ms = []
			bytes_recv = []
			recv_time = []
			for line in f:
				parse = line.split()
				if len(time_ms) > 0 and float(parse[1]) < time_ms[-1]:  # trace error, time not monotonically increasing
					break
				time_ms.append(float(parse[1]))    #time (milliseconds)
				bytes_recv.append(float(parse[4])) #bytes
				recv_time.append(float(parse[5]))  #milliseconds

			time_ms = np.array(time_ms)
			bytes_recv = np.array(bytes_recv)
			recv_time = np.array(recv_time)
			throughput_all = bytes_recv / recv_time

			millisec_time = 0
			mf.write(str(millisec_time) + '\n')

			for i in range(len(throughput_all)):

				throughput = throughput_all[i]
				
				pkt_per_millisec = throughput / BYTES_PER_PKT 

				millisec_count = 0
				pkt_count = 0

				while True:
					millisec_count += 1
					millisec_time += 1
					to_send = (millisec_count * pkt_per_millisec) - pkt_count
					to_send = np.floor(to_send)

					for i in range(int(to_send)):
						mf.write(str(millisec_time) + '\n')

					pkt_count += to_send

					if millisec_count >= recv_time[i]:
						break
                        
def convert_mahimahi_format():
	files = os.listdir(DATA_PATH)

	for f in files:
		file_path = DATA_PATH +  f
		output_path = OUTPUT_PATH + f

		print(file_path)

		with open(file_path, 'r') as f, open(output_path, 'w') as mf:
			time_ms = []
			bytes_recv = []
			recv_time = []
			for line in f:
				parse = line.split()
				if len(time_ms) > 0 and float(parse[0]) < time_ms[-1]:  # trace error, time not monotonically increasing
					break
				time_ms.append(float(parse[0]))
				bytes_recv.append(float(parse[1]))
				recv_time.append(float(parse[2]))

			time_ms = np.array(time_ms)
			bytes_recv = np.array(bytes_recv)
			recv_time = np.array(recv_time)
			throughput_all = bytes_recv / recv_time

			millisec_time = 0
			mf.write(str(millisec_time) + '\n')

			for i in range(len(throughput_all)):

				throughput = throughput_all[i]
				
				pkt_per_millisec = throughput / BYTES_PER_PKT 

				millisec_count = 0
				pkt_count = 0

				while True:
					millisec_count += 1
					millisec_time += 1
					to_send = (millisec_count * pkt_per_millisec) - pkt_count
					to_send = np.floor(to_send)

					for i in range(int(to_send)):
						mf.write(str(millisec_time) + '\n')

					pkt_count += to_send

					if millisec_count >= recv_time[i]:
						break
	
def cut_mahimahi_chunks():
	
	files = os.listdir(DATA_PATH)

	for file in files:
		file_path = DATA_PATH +  file
		output_path = OUTPUT_PATH + file

		print(file_path)

		mahimahi_win = []
		with open(file_path, 'rb') as f:
			for line in f:
				mahimahi_win.append(float(line.split()[0]))

		mahimahi_win = np.array(mahimahi_win)
		chunk = 0
		start_time = 0
		while True:
			end_time = start_time + CHUNK_DURATION

			if end_time * MILLISEC_IN_SEC > np.max(mahimahi_win): 
				break

			print("start time", start_time)

			start_ptr = find_nearest(mahimahi_win, start_time * MILLISEC_IN_SEC)
			end_ptr = find_nearest(mahimahi_win, end_time * MILLISEC_IN_SEC)

			with open(output_path + '_' + str(int(start_time)), 'wb') as f:
				for i in range(start_ptr, end_ptr + 1):
					towrite = mahimahi_win[i] - mahimahi_win[start_ptr]
					f.write(str(int(towrite)) + '\n')

			start_time += CHUNK_JUMP
	
def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

def plot_mahimahi_bandwidth():
	time_all = []
	packet_sent_all = []
	last_time_stamp = 0
	packet_sent = 0
	with open(FILE_MAHIMAHI, 'rb') as f:
		for line in f:
			time_stamp = int(line.split()[0])
			if time_stamp == last_time_stamp:
				packet_sent += 1
				continue
			else:
				time_all.append(last_time_stamp)
				packet_sent_all.append(packet_sent)
				packet_sent = 1
				last_time_stamp = time_stamp

	time_window = np.array(time_all[1:]) - np.array(time_all[:-1])
	throuput_all = PACKET_SIZE * \
				BITS_IN_BYTE * \
				np.array(packet_sent_all[1:]) / \
				time_window * \
				MILLISEC_IN_SEC / \
				MBITS_IN_BITS

	print(throuput_all)
	plt.plot(np.array(time_all[1:]) / MILLISEC_IN_SEC, 
			np.convolve(throuput_all, np.ones(N,)/N, mode='same'))
	plt.xlabel('Time (second)')
	plt.ylabel('Throughput (Mbit/sec)')
	plt.show()

def plot_bandwidth():
	time_ms = []
	bytes_recv = []
	recv_time = []
	with open(FILE_BW, 'rb') as f:
		for line in f:
			parse = line.split()
			time_ms.append(float(parse[1]))
			bytes_recv.append(float(parse[4]))
			recv_time.append(float(parse[5]))
	time_ms = np.array(time_ms)
	bytes_recv = np.array(bytes_recv)
	recv_time = np.array(recv_time)
	throughput_all = bytes_recv / recv_time

	time_ms = time_ms - time_ms[0]
	time_ms = time_ms / MILLISEC_IN_SEC
	throughput_all = throughput_all * BITS_IN_BYTE / MBITS_IN_BITS * MILLISEC_IN_SEC

	plt.plot(time_ms, throughput_all)
	plt.xlabel('Time (second)')
	plt.ylabel('Throughput (Mbit/sec)')
	plt.show()
plot_bandwidth()
