import sys


def get_simple_format():
    input_data_schema = 'fea1,fea2,fea3,fea4,fea5,fea6,fea7,fea8,fea9,label1,label2,label3,label4,label5,label6,label7,label8'.split(',')
    batch_size = 10
    feas = 'fea1,fea2,fea3,fea4,fea5,fea6,fea7,fea8,fea9'.split(',')
    label = 'label2'
    return input_data_schema, feas, batch_size, label

schema, feas, batch_size, label = get_simple_format()
fea_idxs = [ schema.index(col) for col in feas ]

for line in sys.stdin:
    row = line.strip().split('\t')
    for i in fea_idxs:
        row[i] = str(int(row[i])%(1000000000000000))
    print '\t'.join(row)
