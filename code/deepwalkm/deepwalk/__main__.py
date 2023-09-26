#! /usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import os
import sys
import random
from io import open
from argparse import ArgumentParser, FileType, ArgumentDefaultsHelpFormatter
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import logging
from . import graph
from . import walks as serialized_walks
from gensim.models import Word2Vec
from .skipgram import Skipgram
import torch
from six import text_type as unicode
from six import iteritems
from six.moves import range
import psutil
from multiprocessing import cpu_count
import numpy as np
import random
seed = 42
# np.random.seed(seed)  # Numpy module.
# random.seed(seed)
p = psutil.Process(os.getpid())
def set_seed(seed):
    # torch.manual_seed(seed)
    # #进行随机搜索的这个要注释掉
    # # random.seed(seed)
    # np.random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


def debug(type_, value, tb):
  if hasattr(sys, 'ps1') or not sys.stderr.isatty():
    sys.__excepthook__(type_, value, tb)
  else:
    import traceback
    import pdb
    traceback.print_exception(type_, value, tb)
    print(u"\n")
    pdb.pm()


def process(args, adj, ss):

  if args.format == "adjlist":
    G = graph.load_adjacencylist(args.input , adj, ss, undirected=args.undirected)
  elif args.format == "edgelist":
    G = graph.load_edgelist(args.input, undirected=args.undirected)
  elif args.format == "mat":
    G = graph.load_matfile(args.input, variable_name=args.matfile_variable_name, undirected=args.undirected)
  else:
    raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % args.format)

  # print("Number of nodes: {}".format(len(G.nodes())))

  num_walks = len(G.nodes()) * args.number_walks

  # print("Number of walks: {}".format(num_walks))

  data_size = num_walks * args.walk_length

  # print("Data size (walks*length): {}".format(data_size))

  if data_size < args.max_memory_datat_size:
    
    # print("Walking...")
    walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks,
                                        path_length=args.walk_length, alpha=0, rand=random.Random(args.seed))
    # print("Training...")
    # model = Word2Vec(walks, size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1, workers=args.workers)

    # 改正：
    model = Word2Vec(walks, vector_size=args.representation_size, window=args.window_size, min_count=0, sg=1, hs=1,
                     workers=args.workers)


    # print('ddd')
  else:
    print("Data size {} is larger than limit (max-memory-datat-size: {}).  Dumping walks to disk.".format(data_size, args.max_memory_data_size))
    print("Walking...")

    walks_filebase = args.output + ".walks"
    walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=args.number_walks,
                                         path_length=args.walk_length, alpha=0, rand=random.Random(args.seed),
                                         num_workers=args.workers)

    print("Counting vertex frequency...")
    if not args.vertex_freq_degree:
      vertex_counts = serialized_walks.count_textfiles(walk_files, args.workers)
    else:
      # use degree distribution for frequency in tree
      vertex_counts = G.degree(nodes=G.iterkeys())

    print("Training...")
    
    walks_corpus = serialized_walks.WalksCorpus(walk_files)
    model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                     size=args.representation_size,
                     window=args.window_size, min_count=0, trim_rule=None, workers=args.workers, rand=random.Random(args.seed))
    
  model.wv.save_word2vec_format(args.output)
  dat = np.zeros((ss, args.representation_size))
  with open(args.output, 'r') as f:
      files = f.readlines()
      for index, ff in enumerate(files):
          if index == 0:
            continue
          shuzu = ff.strip().split(' ')
          shuzu = list(map(float,shuzu))
          # print(shuzu)
          # exit()
          dat[int(shuzu[0])] = shuzu[1:]
  return dat
  # print(model.wv.vector_size)
  # print(model.wv )
  # exit()


def get_embedding(adj, ss, hid_size):
  parser = ArgumentParser("deepwalk",
                          formatter_class=ArgumentDefaultsHelpFormatter,
                          conflict_handler='resolve')

  parser.add_argument("--debug", dest="debug", action='store_true', default=False,
                      help="drop a debugger if an exception is raised.")

  parser.add_argument('--format', default='adjlist',
                      help='File format of input file')

  parser.add_argument('--input', nargs='?', default= 'gcn.adjlist',
                      help='Input graph file')

  parser.add_argument("-l", "--log", dest="log", default="INFO",
                      help="log verbosity level")

  parser.add_argument('--matfile-variable-name', default='network',
                      help='variable name of adjacency matrix inside a .mat file.')

  parser.add_argument('--max-memory-datat-size', default=1000000000, type=int,
                      help='Size to start dumping walks to disk, instead of keeping them in memory.')

  parser.add_argument('--number-walks', default=10, type=int,
                      help='Number of random walks to start at each node')

  parser.add_argument('--output', default='cirrna_drug.embeddings',
                      help='Output representation file')

  parser.add_argument('--representation-size', default=hid_size, type=int,
                      help='Number of latent dimensions to learn for each node.')

  parser.add_argument('--seed', default=1, type=int,
                      help='Seed for random walk generator.')

  parser.add_argument('--undirected', default=True, type=bool,
                      help='Treat graph as undirected.')

  parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                      help='Use vertex degree to estimate the frequency of nodes '
                           'in the random walks. This option is faster than '
                           'calculating the vocabulary.')

  parser.add_argument('--walk-length', default=30, type=int,
                      help='Length of the random walk started at each node')

  parser.add_argument('--window-size', default=5, type=int,
                      help='Window size of skipgram model.')

  parser.add_argument('--workers', default=1, type=int,
                      help='Number of parallel processes.')


  args = parser.parse_args()
  numeric_level = getattr(logging, args.log.upper(), None)
  logging.basicConfig(format=LOGFORMAT)
  logger.setLevel(numeric_level)

  if args.debug:
   sys.excepthook = debug

  dat = process(args, adj,ss)
  # print('jidjifdjfiidijd')
  return dat
'''
if __name__ == "__main__":
  sys.exit(main())
'''