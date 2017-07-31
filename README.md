## Introduction

PR4NMT provides a general framework to incorporate multiple, arbitrary prior knowledge into Neural Machine Translation. Please refer to the following paper for details:

> Jiacheng Zhang, Yang Liu, Huanbo Luan, Jingfang Xu and Maosong Sun. 2017. [Prior Knowledge Integration for Neural Machine Translation using Posterior Regularization](http://nlp.csai.tsinghua.edu.cn/~ly/papers/acl2017_zjc.pdf). In Proceedings of ACL 2017, Vancouver, Canada, July.

## Installation

PR4NMT is built on top of [THUMT](http://github.com/thumt/THUMT). It requires THEANO 0.8.2 or above version (0.8.2 is recommended)

`` pip install theano==0.8.2 ``

## Preparation

Firstly, modify THUMT.config in the config directory to specify features and hyper-parameters. if bilingual dictionary (or phrase table) feature is selected, use cPickle to stringify a bilingual dictionary (or phrase table) in the following format:

``` 
import cPickle

word_table = [[source word 1, target word 1] , [source word 2, target word 2], ...] 
cPickle.dump(word_table, open('word_table', 'w'))
```

## Training

The trainer.py script in the scripts folder is used for training NMT models. We recommend initializing MRT with the best model output by MLE using the --init-model-file option. The command for running PR is given by:

```
python /Users/Jack/THUMT/scripts/trainer.py --config-file /Users/Jack/THUMT/config/THUMT.config --trn-src-file /Users/Jack/THUMT/data/train.src --trn-trg-file /Users/Jack/THUMT/data/train.trg --vld-src-file /Users/Jack/THUMT/data/valid.src --vld-trg-file /Users/Jack/THUMT/data/valid.trg --training-criterion 3 --word-table-file /Users/Jack/THUMT/data/word_table --phrase-table-file /Users/Jack/THUMT/data/phrase_table --init-model-file model_mle_best.npz --device gpu0 
```

Noting that the effect of PR4NMT is sensitive to MLE weight and PR weight.

## Decoding

Given a trained model model_best.npz, please run the following command to translate the test set without evaluation:

```
python /Users/Jack/THUMT/scripts/test.py --model-file model_best.npz --test-src-file /Users/Jack/THUMT/data/test.src --test-trg-file test.trans --device gpu0
```

## License

The source code is dual licensed. Open source licensing is under the [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause), which allows free use for research purposes. For commercial licensing, please email [thumt17@gmail.com](mailto:thumt17@gmail.com).
