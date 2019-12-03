import os

def get_csv():

    trial = 'R1'
    cwd = os.getcwd()
    path = os.path.join(cwd, r'eeg_dataset/')
    path = os.path.join(path, trial)
    trains = []

    for data_directory in os.listdir(path):
        data_name = os.path.join(path, data_directory)
        trains.append(data_name)

    with open(r'csv/'+trial+'.csv', 'w') as output:
        for data_name in trains:
            output.write("%s" % data_name)
            output.write("\n")
def main():
    get_csv()


if __name__ == '__main__':
    main()
