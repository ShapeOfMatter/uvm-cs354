from collections import defaultdict
from os import link, mkdir, walk
from os.path import join as pjoin
from subprocess import run

files = {dirpath.split('/')[1]: tuple((dirpath, filename, int(filename.split('.')[0]))
                                      for filename in filenames)
         for (dirpath, dirnames, filenames)
         in walk('sketch_all_data')
         if '.DS_Store' not in filenames}

for (category, tuples) in files.items():
    in_order = sorted(tuples, key=lambda t: t[2])
    for (newpath, selection) in [(pjoin('sketch_train', category), in_order[:60]),
                                 (pjoin('sketch_test', category), in_order[60:])]:   
        mkdir(newpath)
        for (dirpath, filename, index) in selection:
            link(pjoin(dirpath, filename), pjoin(newpath, filename))
    print(f'{category}: {len(in_order)}')


