import argparse

def read_argv():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', help='Enable the debug log', action='store_true')
    parser.add_argument('-i', '--input', help='input file', type=argparse.FileType('r'), required=True)
    parser.add_argument('-o', '--output', help='input output file', type=argparse.FileType('w'))
    parser.add_argument('-k', '--keyword', help='Keyword to filter lines', type=str)
    
    args = parser.parse_args()
    return args

args = read_argv()
input_file = args.input
keyword = args.keyword
output_file = args.output

lines = input_file.readlines()
keys = {}

for line in lines:
    if keyword and keyword not in line:
        continue
    if line not in keys:
        keys[line] = 0
    keys[line] += 1

output_lines = [f'{count},{line}' for line, count in keys.items()]

# Print to console
for line in output_lines:
    print(line, end='')

# Write to output file if provided
if output_file:
    output_file.writelines(output_lines)
    output_file.close()
