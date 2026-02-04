import numpy as np
import os
from evaluator.evaluator_cross_city import Evaluator as CE


if __name__ == '__main__':
    st_pair = 'cd-xa'
    s_dataset = 'cd'
    t_dataset = 'xa'

    s_emb = np.load('./save_emb/{}/{}_region_emb.npy'.format(
        st_pair,  s_dataset))
    t_emb = np.load('./save_emb/{}/{}_region_emb.npy'.format(
        st_pair, t_dataset))

    evaluator = CE(exp_id=None)
    ce_result = evaluator.evaluate(s_dataset, t_dataset, s_emb, t_emb)

