"""Redraw thesis summaries from frozen inputs, never from rounded prose."""
from pathlib import Path
import csv
import hashlib
import json
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, NullLocator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIG = HERE / 'thesis/figures/main'
SRC = HERE / 'summary_figure_sources'
SRC.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 14,
                     'axes.spines.top': False, 'axes.spines.right': False})
BLUE, TEAL, GREY = '#173f5f', '#007f82', '#66717c'


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save(fig, name, inputs, facts):
    fig.savefig(FIG / (name + '.png'), dpi=300, facecolor='white')
    plt.close(fig)
    record = {'inputs': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in inputs}, 'facts': facts,
              'generator': str(Path(__file__).relative_to(ROOT))}
    (SRC / (name + '.json')).write_text(json.dumps(record, indent=2), encoding='utf-8')


def cost():
    source = ROOT / 'total_cost_results/cost_aggregate.csv'
    data = rows(source)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8.2))
    for col, (ds, title) in enumerate([('fashion', 'Fashion-MNIST'),
                                      ('cifar10', 'CIFAR-10'), ('sign', 'Sign')]):
        for row, (field, err, label) in enumerate([
                ('mean_total_seconds_among_reached', 'ci95_total_seconds_among_reached', 'Time to target (s)'),
                ('mean_net_joules_among_reached', 'ci95_net_joules_among_reached', 'Estimated net GPU energy (J)')]):
            ax = axes[row, col]
            for method, color, label_method, marker in [('kaiming', BLUE, 'Kaiming', 'o'),
                                                        ('lowrank_wmf', TEAL, 'Low-rank DI-WMF', 'D')]:
                group = sorted([r for r in data if r['dataset'] == ds and r['method'] == method
                                and int(r['reached_runs']) > 0], key=lambda r: float(r['threshold_accuracy']))
                ax.errorbar([float(r['threshold_accuracy']) for r in group],
                            [float(r[field]) for r in group], yerr=[float(r[err]) for r in group],
                            color=color, marker=marker, capsize=4, lw=1.8, label=label_method)
                if ds == 'sign' and method == 'kaiming':
                    r = next(r for r in group if float(r['threshold_accuracy']) == 40)
                    ax.annotate('2/5 reached', (40, float(r[field])), xytext=(12, 14),
                                textcoords='offset points', ha='left', fontsize=12, color=BLUE,
                                bbox={'facecolor':'white', 'edgecolor':'none', 'pad':1})
            if row == 0:
                ax.set_title(title, fontweight='bold', pad=14)
            ax.set_xlabel('Validation accuracy\ntarget (%)')
            if col == 0:
                ax.set_ylabel(label)
            ticks = sorted({float(r['threshold_accuracy']) for r in data if r['dataset'] == ds})
            ax.set_xticks(ticks)
            ax.yaxis.set_major_locator(MaxNLocator(5))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.yaxis.set_minor_locator(NullLocator())
            ax.grid(axis='y', color='#dbe2e8', linewidth=.65)
            ax.margins(x=.13, y=.22)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .938), ncol=2, frameon=False)
    fig.suptitle('Cost to selected targets: means among runs that reached them', y=.988,
                 fontsize=16, fontweight='bold')
    fig.text(.5, .047, 'Full data | 20 epochs | 5 seeds | intervals: descriptive 95% t CIs', ha='center', fontsize=12)
    fig.text(.5, .016, 'Energy uses whole-run mean-power prorating, not a measured prefix integral. '
             'Sign Kaiming never reached 60%.', ha='center', fontsize=11, color=GREY)
    fig.subplots_adjust(left=.11, right=.985, bottom=.18, top=.84, hspace=.68, wspace=.36)
    save(fig, 'F11_time_energy_to_accuracy', [source],
         {'estimand': 'among reached runs', 'energy': 'E_init + t_hit/t_train * E_train',
          'Sign_Kaiming_reached_40': '2/5', 'Sign_Kaiming_reached_60': '0/5'})


def summary():
    source = ROOT / 'submission_control_results/final_main_ten_seed_registry.csv'
    data = rows(source)
    assert len(data) == 9 and all(int(r['paired_seeds']) == 10 for r in data)
    lookup = {(r['dataset'], r['metric']): r for r in data}
    fig = plt.figure(figsize=(12.6, 9.8))
    fig.text(.05, .95, 'What the study supports', fontsize=21, fontweight='bold', color=BLUE)
    fig.text(.05, .91, 'Complete initializer: low-rank DI-WMF minus Kaiming '
             '| full data | 50 epochs | 10 paired seeds', fontsize=13)
    ax = fig.add_axes([.04, .52, .92, .34])
    ax.axis('off')
    dataset_labels = [('fashion', 'Fashion-MNIST'), ('cifar10', 'CIFAR-10'), ('sign', 'Sign')]
    xs = [.40, .63, .86]
    for x, (_, title) in zip(xs, dataset_labels):
        ax.text(x, .93, title, ha='center', fontsize=16, fontweight='bold')
    facts = {}
    for y, metric, label in [(.71, 'Epoch-0', 'Epoch-0 accuracy'),
                             (.44, 'Post-update AULC', 'Post-update AULC 1:50'),
                             (.17, 'Final test', 'Final test accuracy')]:
        ax.text(.01, y, label, va='center', fontsize=14.5, fontweight='bold')
        for x, (ds, _) in zip(xs, dataset_labels):
            r = lookup[ds, metric]
            effect, lo, hi = (float(r[k]) for k in ('mean_paired_difference', 'ci95_low', 'ci95_high'))
            color = BLUE if metric == 'Epoch-0' else (TEAL if lo > 0 else GREY)
            ax.text(x, y+.032, f'{effect:+.2f}', ha='center', fontsize=25, fontweight='bold', color=color)
            ax.text(x, y-.085, f'[{lo:.2f}, {hi:.2f}]', ha='center', fontsize=14.5, color=GREY)
            facts[ds + '/' + metric] = {'effect': effect, 'ci_low': lo, 'ci_high': hi}
    ax.text(.01, -.06, 'Percentage-point effects; brackets are unadjusted paired 95% confidence intervals.',
            fontsize=12, color=GREY)
    items = [
        ('The classifier explains most of Epoch 0',
         'The fitted-head control tests two representations with the same head-fitting rule. '
         'Fashion-MNIST and Sign retain post-update gains under the broader 18-test Holm check.'),
        ('Transfer has explicit limits',
         'Corruption retention is mixed; ordinary CNNs do not inherit rotation invariance. '
         'Template provenance is traceable, but it is not proof of human-semantic meaning.'),
        ('Physiology: a separate, corrected patient-level evaluation',
         '46 held-out patients. Manual-reference RR MAE: 1.625 bpm for pretrained Correncoder '
         'and 3.334 bpm for band-pass FFT. The paired mean difference favours Correncoder, '
         'but does not survive the 120-test Holm correction (p = 0.100).')]
    for y, (title, body) in zip([.42, .285, .15], items):
        fig.text(.06, y, title, fontsize=16, fontweight='bold', color=BLUE)
        fig.text(.06, y-.028, '\n'.join(textwrap.wrap(body, width=115)), va='top', fontsize=14.5,
                 linespacing=1.28)
    phys = HERE / 'physiology_evaluation/physiology_summary_patient46.csv'
    save(fig, 'F20_cross_task_conclusion_map', [source, phys], facts)


if __name__ == '__main__':
    cost()
    summary()
    print('F11 and F20 redrawn from frozen result sources.')
