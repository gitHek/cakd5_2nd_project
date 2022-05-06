import os
import argparse

def data_split(file_path,output_dir):
    with open(f'{file_path}/seq.in','r') as f:
          input = f.readlines()
    with open(f'{file_path}/seq.out','r') as f:
          output = f.readlines()
          
    train_input = input[:8000]
    val_input = input[8000:9000]
    test_input = input[9000:10000]
    train_output = output[:8000]
    val_output = output[8000:9000]
    test_output = output[9000:10000]

    base_dir = f'{output_dir}'

    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)
    validation_dir = os.path.join(base_dir, 'validation')
    os.mkdir(validation_dir)
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)

    with open(f'{train_dir}/seq.in','w') as f:
        for i in train_input:
            f.write(i)
    with open(f'{train_dir}/seq.out','w') as f:
        for i in train_output:
            f.write(i)
    with open(f'{validation_dir}/seq.in','w') as f:
        for i in val_input:
            f.write(i)
    with open(f'{validation_dir}/seq.out','w') as f:
        for i in val_output:
            f.write(i)
    with open(f'{test_dir}/seq.in','w') as f:
        for i in test_input:
            f.write(i)
    with open(f'{test_dir}/seq.out','w') as f:
        for i in test_output:
            f.write(i)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", "-i", help = "seq file path", type = str, required=True)
    parser.add_argument("--output_dir", "-o", help = "output_dir", type = str, required=True)
        
    args = parser.parse_args()
    file_path = args.file_path
    output_dir = args.output_dir

    data_split(file_path, output_dir)
    