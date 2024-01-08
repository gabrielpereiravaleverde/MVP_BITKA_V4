import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import optuna

sns.set_style('darkgrid')
# sns.set_context('talk')
sns.set_palette('rainbow')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

class OptunaResults:
    def __init__(self, study, params):
        self.study = study
        self.params = params
        self.__to_df()

    def __to_df(self):
        param_results = {'target' : [-1* t.value for t in self.study.trials], 'trial' : range(len(self.study.trials))}
        for k in self.params:
            param_results[k] = [t.params[k] for t in self.study.trials]
        self.param_results = pd.DataFrame(param_results)

    def plot_param(self, figsize = (20,6)):
        fig, ax = plt.subplots(1, len(self.params), figsize = figsize, sharey = True)
        for i, c in enumerate(self.params):
            sns.scatterplot(x = c, y = 'target', data = self.param_results, hue = 'trial', ax = ax[i])
            ax[i].set_title(c)
            ax[i].set_xlabel("")
            ax[i].get_legend().remove()
            if i == 0:
                ax[i].set_ylabel('Target')
            elif i == len(self.params) - 1:
                ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))  

class SimulationResults:
    def __init__(self, results, params):
        self.params = params
        data = []
        self.studies = [x.study for x in results['searcher_state']]
        self.general_results = pd.DataFrame(results)
        self.baseline_history = pd.concat([x.assign(context = i) for i,x in enumerate(results['baseline_state'])], ignore_index = True)
        self.baseline_history['fitness'] *= -1
        
        for i, study in enumerate(self.studies):
            data.append(OptunaResults(study, params).param_results.assign(context = i))
        self.results = pd.concat(data, ignore_index = True)
        self.baseline_name = results['baseline_name'][0]
        self.searcher_name = results['searcher_name'][0]
        save_cols = params + ['fitness']

        comp_data = [self.general_results.rename(columns = {'target' : 'fitness'})[save_cols].assign(mode = 'Histórico')]
        for mode in [self.searcher_name] + [self.baseline_name]:
            comp_data.append(self.general_results[[mode + "_" + x for x in save_cols]].rename(columns = dict(zip([mode + "_" + x for x in save_cols], save_cols))).assign(mode = mode))

        self.comp_data = pd.concat(comp_data, ignore_index = True)

    def full_plot(self, figsize = (20, 10)):
        fig, ax = plt.subplots(3, 3, figsize = figsize)
        num_bins = 50
        self.baseline_history['scaled'] = self.baseline_history.groupby(['context', 'dv'])['fitness'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

        for i, c in enumerate(self.params):
            self.results['bin'] = pd.cut(self.results[c], bins = num_bins).apply(lambda x: x.mid)
            g = self.results.groupby(['context', 'bin'])['target'].max().reset_index()
            g['scaled'] = g.groupby('context')['target'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

            sns.lineplot(x = 'aux_bin', y = 'scaled', hue = 'context', data = self.baseline_history[self.baseline_history['dv'] == c], ax = ax[0, i], marker = 'o')
            sns.lineplot(x = 'bin', y = 'scaled', hue = 'context', data = g, ax = ax[1, i], marker = 'o')
            sns.boxplot(x = 'mode', y = c, data = self.comp_data, ax = ax[2,i])

            ax[0,i].set_title(c)
            ax[2, i].set_ylabel('')
            for j in [0,1,2]:
                ax[j, i].set_xlabel("")
                ax[j, i].set_ylabel("")
                leg = ax[j, i].get_legend()
                if leg: 
                    leg.remove()
                if i == len(self.params) - 1:
                    ax[j, i].legend(loc='center left', bbox_to_anchor=(1, 0.5))  

        ax[0,0].set_ylabel(self.baseline_name)
        ax[1,0].set_ylabel(self.searcher_name)

    def plot_param(self, figsize = (20,6)):
        fig, ax = plt.subplots(1, 3, figsize = (20,6), sharey = True)
        num_bins = 50
        for i, c in enumerate(self.params):
            self.results['bin'] = pd.cut(self.results[c], bins = num_bins).apply(lambda x: x.mid)
            g = self.results.groupby(['context', 'bin'])['target'].max().reset_index()
            g['scaled'] = g.groupby('context')['target'].transform(lambda x: (x - x.min()) / (x.max() - x.min()))
            sns.lineplot(x = 'bin', y = 'scaled', hue = 'context', data = g, ax = ax[i], marker = 'o')
            ax[i].set_title(c)
            ax[i].set_xlabel("")
            ax[i].get_legend().remove()
            if i == 0:
                ax[i].set_ylabel('Target')
            elif i == len(self.params) - 1:
                ax[i].legend(loc='center left', bbox_to_anchor=(1, 0.5))  
    
    def plot_decision_variable(self):
        fig, ax = plt.subplots(1, 3, figsize = (20,6))
        for i, c in enumerate(self.params):
            sns.boxplot(x = 'mode', y = c, data = self.comp_data, ax = ax[i])
            ax[i].set_title(c)
            ax[i].set_xlabel("")
            ax[i].set_ylabel("")

    def searcher_superiority(self):
        return np.mean(self.general_results['searcher_fitness'] < self.general_results['baseline_fitness'])

    def importances(self):
        self.importances = {k:[] for k in self.params}
        for s in self.studies:
            imp = optuna.importance.get_param_importances(s)
            for k, v in imp.items():
                self.importances[k].append(v)