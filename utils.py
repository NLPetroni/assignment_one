# ! pip install wandb # colab only

import torch

PAD_TOKEN = 400000

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

classes = {'$', 'NN', ',', 'RBS', 'FW', 'CC', '#', 'VBD', 'PRP', 'RBR', 'LS', ':', 'VBZ', 'MD',
           'EX', 'RB', 'WRB', 'NNS', 'VBG', 'PRP$', 'JJR', 'WP$', 'WP', '-LRB-', 'WDT', '``',
           '.', 'CD', 'JJ', "''", 'UH', 'VBN', 'IN', 'SYM', 'DT', 'JJS', '-RRB-', 'RP', 'VB',
           'POS', 'NNP', 'PDT', 'NNPS', 'VBP', 'TO', '<PAD>'}
punctuation_cls = {'$', ',', '#', ':', '-LRB-', '``', '.', "''", 'SYM', '-RRB-', '<PAD>'}
class2idx = {c: i for i, c in enumerate(classes)}


def download_and_unzip(url, save_dir='.'):
    # downloads and unzips url, if not already downloaded
    # used for downloading dataset and glove embeddings
    import os
    from urllib.request import urlopen
    from io import BytesIO
    from zipfile import ZipFile
    fname = url.split('/')[-1][:-4] if save_dir == '.' else save_dir
    if fname not in os.listdir():
        print(f'downloading and unzipping {fname}...', end=' ')
        r = urlopen(url)
        zipf = ZipFile(BytesIO(r.read()))
        zipf.extractall(path=save_dir)
        print(f'completed')

def get_wandbkey():
    with open('wandbkey.txt') as f:
        return f.read().strip()
