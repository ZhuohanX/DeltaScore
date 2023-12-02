import argparse, sys, json
import torch.cuda
from scipy.stats import pearsonr, spearmanr, kendalltau, ttest_ind
from eva.bartscore import BARTScorer
from eva.optscore import OPTScorer
from eva.flant5score import FLANT5Scorer
from eva.bloomscore import BLOOMScorer
from eva.templates.dialogue import DialogueTemplates
from eva.perturb.perturb import consistency
from tqdm import tqdm
import numpy as np

def printcorrelations(scores, annotations):
    print(f"Pearson: {round(pearsonr(annotations['fluency'],scores)[0]*100, 1)} & {round(pearsonr(annotations['coherence'],scores)[0]*100, 1)} & {round(pearsonr(annotations['relatedness'],scores)[0]*100, 1)} & {round(pearsonr(annotations['logicality'],scores)[0]*100, 1)} & {round(pearsonr(annotations['interestingness'],scores)[0]*100, 1)}")
    print(f"Spearman: {round(spearmanr(annotations['fluency'], scores)[0]*100, 1)} & {round(spearmanr(annotations['coherence'], scores)[0]*100, 1)} & {round(spearmanr(annotations['relatedness'], scores)[0]*100, 1)} & {round(spearmanr(annotations['logicality'], scores)[0]*100, 1)} & {round(spearmanr(annotations['interestingness'], scores)[0]*100, 1)}")
    print(f"Kendall: {round(kendalltau(annotations['fluency'], scores)[0]*100, 1)} & {round(kendalltau(annotations['coherence'], scores)[0]*100, 1)} & {round(kendalltau(annotations['relatedness'], scores)[0]*100, 1)} & {round(kendalltau(annotations['logicality'], scores)[0]*100, 1)} & {round(kendalltau(annotations['interestingness'], scores)[0]*100, 1)}")

def load(data_dir, data_name):
    with open(f"./data/{data_dir}/{data_name}.jsonl", "r") as fin:

        annotations = {
            'fluency': [],
            'coherence': [],
            'relatedness': [],
            'logicality': [],
            'interestingness': []
        }
        titles = []
        stories = []
        references = []
        ptd_stories = []

        for line in tqdm(fin.readlines()):
            data = json.loads(line.strip())
            titles.append(data['title'])
            stories.append(data['story'])
            references.append(data['human'])
            annotations['fluency'].append(data['fluency'])
            annotations['coherence'].append(data['coherence'])
            annotations['relatedness'].append(data['relatedness'])
            annotations['logicality'].append(data['logicality'])
            annotations['interestingness'].append(data['interestingness'])
        return titles, stories, references, ptd_stories, annotations

def perturb_stories(stories, perturbation, ratio):
    dg = DialogueTemplates()
    con = consistency('story')

    if perturbation == 'jumble':
        ptd_stories = [dg.jumble(st, ratio) for st in stories]
    elif perturbation == 'typos':
        ptd_stories = [dg.typos(st, ratio) for st in stories]
    elif perturbation == 'antonym':
        ptd_stories = [con.substitute_antonym(st, ratio) for st in stories]
    elif perturbation == 'add_negation':
        ptd_stories = [dg.add_negation(st) for st in stories]
    return ptd_stories


def main(args):

    print(torch.cuda.device_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.metric == 'bart':
        metric = BARTScorer(device=device, checkpoint=args.checkpoint)
    elif args.metric == 'opt':
        metric = OPTScorer(device=device, checkpoint=args.checkpoint)
    elif args.metric == 'gpt':
        metric = GPTScorer(device=device, checkpoint=args.checkpoint)
    elif args.metric == 't5':
        metric = FLANT5Scorer(device=device, checkpoint=args.checkpoint)
    elif args.metric == 'bloom':
        metric = BLOOMScorer(device=device, checkpoint=args.checkpoint)


    data_dir = 'crowdsource'
    perturbation = args.perturbation
    for data_name in ['roc', 'wp']:
        titles, stories, references, ptd_stories, annotations = load(data_dir, data_name)
        for ratio in np.arange(0.1, 1.1, 0.1):
            if perturbation in ['jumble', 'typos', 'antonym']:
                for i in range(5):
                    ptd_stories = perturb_stories(stories, perturbation, ratio)
                    if i == 0:
                        ptd_scores = metric.score(titles, ptd_stories, batch_size=1)
                    else:
                        ptd_scores = [(x+y)for x,y in zip(ptd_scores, metric.score(titles, ptd_stories, batch_size=1))]
                ptd_scores = [x/5.0 for x in ptd_scores]
            else:
                ptd_stories = perturb_stories(stories, perturbation, ratio)
                ptd_scores = metric.score(titles, ptd_stories, batch_size=1)

            ori_scores = metric.score(titles, stories, batch_size=1)
            delta_scores = [(x-y) for x,y in zip(ori_scores, ptd_scores)]

            new_datas = []
            with open(f"./data/{data_dir}/{data_name}.jsonl", "r") as fin:
                for i, line in enumerate(fin.readlines()):
                    data = json.loads(line.strip())
                    data[args.metric] = ori_scores[i]
                    data[f'{args.metric}_{perturbation}_{ratio}'] = delta_scores[i]
                    new_datas.append(data)
            fw = open(f"./data/{data_dir}/{data_name}_{args.metric}_{perturbation}_{ratio}.jsonl", 'w')
            for data in new_datas:
                json.dump(data, fw)
                fw.write('\n')
            fw.close()

            print(f'=================={data_dir} {data_name} {args.metric} {args.checkpoint}=====================')
            print(f'==================Original Scores:=====================')
            printcorrelations(ori_scores, annotations)
            print(f'==================Delta Scores with {perturbation} {ratio}:=====================')
            printcorrelations(delta_scores, annotations)
            print(f'=================={data_dir} {data_name} {args.metric} {args.checkpoint}=====================')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', type=str, default='opt', help='metric')
    parser.add_argument('--checkpoint', type=str, default='facebook/opt-350m', help='checkpoint')
    parser.add_argument('--perturbation', type=str, default='jumble', help='perturbation')
    parser.add_argument('--ratio', type=float, default=0.0, help='ratio')
    args = parser.parse_args()
    main(args)