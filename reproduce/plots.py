#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Andrey Chesnokov
"""Draw the paper's three experimental figures from recorded exact values.

Install matplotlib separately. This renders recorded observations; run.py
produces fresh measurements. Each exported point includes its source key.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'bttb-plot-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

HERE = Path(__file__).resolve().parent
COLORS = ['#0072B2', '#D55E00', '#009E73']


def draw(values_file, output):
    records = {r['key']: r for r in csv.DictReader(values_file.open())}
    output.mkdir(parents=True, exist_ok=True)
    points = []
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10,
                         'mathtext.fontset': 'cm', 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42,
                         'svg.fonttype': 'none', 'lines.linewidth': 1.6})

    def series(figure, label, xs, keys, scale=1):
        ys = []
        for x, key in zip(xs, keys):
            row = records[key]
            value = float(row['value'].replace(',', ''))
            ys.append(value * scale)
            points.append({'figure': figure, 'series': label, 'x': x,
                           'value': value, 'unit': row['unit'], 'source_key': key,
                           'plotted_value': value * scale})
        return ys

    def axis(ax, xs, logx=False, logy=True):
        if logx:
            ax.set_xscale('log', base=2)
        if logy:
            ax.set_yscale('log')
        ax.xaxis.set_major_locator(FixedLocator(xs))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(axis='y', which='major', color='#dddddd', linewidth=.6)
        ax.set_axisbelow(True)

    def save(fig, name):
        fig.savefig(output / (name + '.pdf'), bbox_inches='tight')
        fig.savefig(output / (name + '.svg'), bbox_inches='tight')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.1), layout='constrained')
    for R, xs, color, marker, linestyle in [
            (1, [1, 2, 4, 8, 16, 32, 48, 63], COLORS[0], 'o', '-'),
            (4, [2, 4, 8, 16, 32, 48, 63], COLORS[1], 's', '--')]:
        label = f'$R={R}$'
        ys = series('bandwidth', label, xs,
                    [f'cc.n128.k{k}.r{R}.cc_r.speedup_dense_lu' for k in xs])
        ax.plot(xs, ys, linestyle, marker=marker, markersize=4.5, color=color, label=label)
    axis(ax, [1, 2, 4, 8, 16, 32, 63], logx=True)
    ax.axhline(1, color='#555555', linewidth=1, linestyle=':')
    ax.axvspan(16, 32, color='#777777', alpha=.10, zorder=0)
    ax.set_xlabel('Block half-bandwidth $k$')
    ax.set_ylabel('Speedup over dense LU')
    ax.set_ylim(.09, 130)
    ax.text(.02, .05, '$n=128$', transform=ax.transAxes, va='bottom')
    ax.legend(loc='upper right', frameon=False)
    save(fig, 'bandwidth')

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True, layout='constrained')
    ns = [448, 512, 896, 1024]
    for ax, R in zip(axes, [1, 4]):
        for k, color, marker in zip([1, 2, 4], COLORS, ['o', 's', '^']):
            label = f'CC-R, $k={k}$'
            ys = series('scaling', f'{label}, R={R}', ns,
                        [f'cc.n{n}.k{k}.r{R}.cc_r.cpu' for n in ns], scale=1/60)
            ax.plot(ns, ys, marker=marker, markersize=4, color=color, label=label)
        if R == 4:
            label = 'CC, $k=1$'
            ys = series('scaling', label + ', R=4', ns,
                        [f'cc.n{n}.k1.r4.cc.cpu' for n in ns], scale=1/60)
            ax.plot(ns, ys, '--', marker='x', color='#333333', label=label)
        axis(ax, ns, logx=True)
        ax.tick_params(axis='x', labelsize=9)
        ax.set_xlabel('Block order $n$')
        ax.set_title(f'$R={R}$', fontsize=11)
        ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    axes[0].set_ylabel('Total CPU time (min)')
    axes[0].yaxis.set_major_locator(FixedLocator([5, 10, 30, 60, 180]))
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    save(fig, 'scaling')

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), layout='constrained')
    ax = axes[0]
    ns = [16, 32, 48, 64, 96, 128, 160, 192]
    for method, label, color, marker in [('hir', 'HIR', COLORS[0], 'o'),
                                         ('hsc_q0', 'HSC-q0', COLORS[1], 's')]:
        ys = series('halfstrip', label, ns,
                    [f'hs.n{n}.w2.r1.{method}.speedup_dense_lu' for n in ns])
        ax.plot(ns, ys, marker=marker, markersize=4, color=color, label=label)
    axis(ax, [16, 48, 96, 160, 192])
    ax.set_title('Invertible outermost block', fontsize=10.5)
    ax.set_xlabel('Block order $n$')
    ax.set_ylabel('Speedup over dense LU')
    ax.legend(frameon=False, loc='upper left')
    ax = axes[1]
    Rs = [1, 4, 16, 64]
    for n, color, marker in zip([160, 192], COLORS, ['o', 's']):
        label = f'HPR, $n={n}$'
        ys = series('halfstrip', label, Rs,
                    [f'fh0.n{n}.w2.r{R}.hpr.speedup_dense_lu' for R in Rs])
        ax.plot(Rs, ys, marker=marker, markersize=4, color=color, label=label)
    axis(ax, Rs, logx=True)
    ax.set_title('Singular outermost block', fontsize=10.5)
    ax.set_xlabel('Forward/transpose pairs $R$')
    ax.legend(frameon=False, loc='lower left')
    ax.set_ylim(.20, 1.12)
    ax.yaxis.set_major_locator(FixedLocator([.2, .4, .6, .8, 1]))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    ax.yaxis.set_minor_formatter(NullFormatter())
    for ax in axes:
        ax.axhline(1, color='#555555', linewidth=1, linestyle=':')
    save(fig, 'halfstrip')

    (output / 'plot_points.json').write_text(json.dumps(points, indent=2) + '\n')
    print(f'Drew three recorded-data figures ({len(points)} points) in {output}.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--values', type=Path, default=(HERE / 'reference/paper_numbers.csv' if (HERE / 'reference/paper_numbers.csv').exists() else HERE.parent / 'data/paper_plots/recorded.csv'))
    parser.add_argument('--output', type=Path, default=HERE / 'output/plots')
    args = parser.parse_args()
    draw(args.values, args.output)


if __name__ == '__main__':
    main()
