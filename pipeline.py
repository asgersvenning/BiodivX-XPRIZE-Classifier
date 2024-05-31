### Placeholder for orchestrating the execution pipeline ###
# We will run each of the dependency scripts as a subprocess or os.system
# Script will also handle input-output formatting

import zipfile, argparse

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('-i', '--input_zip_path', type=str, help='Path to the input zip file', required=True)
    args_parser.add_argument('-o', '--output_zip_path', type=str, help='Path to the output zip file', default='output.zip', required=False)
    args = args_parser.parse_args()

    input_zip_path = args.input_zip_path
    output_zip_path = args.output_zip_path

    # Read the input zip file contents 
    input_content = []
    with zipfile.ZipFile(input_zip_path, 'r') as z:
        for file in z.filelist:
            if not file.filename.endswith('.txt'):
                continue
            with z.open(file.filename, 'r') as f:
                for line in f:
                    input_content.append(line.decode('utf-8').strip())

    # Write the output zip file contents
    with zipfile.ZipFile(output_zip_path, 'w') as z:
        with z.open('output.txt', 'w') as f:
            f.write('\n'.join(input_content).encode('utf-8'))

