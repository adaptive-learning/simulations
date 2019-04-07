from copy import deepcopy
import json
from math import log
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import seaborn as sns
from tqdm import tqdm


class Item(object):
    def __init__(self, id, b, a=-1):
        self.id = id
        self.b = b
        self.a = a
        self.estimated_b = 0


class Student(object):
    def __init__(self, id, skill):
        self.id = id
        self.skill = skill
        self.solved_items = 0


class Simulation(object):
    def __init__(self, setting):
        self.students = None
        self.items = None
        self.true_b = None
        self.means = None
        self.available_items = None
        self.active_student = None
        self.active_item = None
        self.item_order = None
        self.remaining_solve_time = None
        self.solve_time = None
        self.log = None
        self.order_log = None
        self.random_students_id = []
        self.setting = deepcopy(setting)

    def _chain_executor(self, chain):
        for method_name in chain:
            getattr(self, method_name)()

    def _chain_executor_cumulative(self, chain):
        result = []
        for method_name in chain:
            result = getattr(self, method_name)(result)
        return result

    def _set_seed(self):
        random.seed(self.setting['seed'])
        np.random.seed(self.setting['seed'])

    def run(self):
        log_id = 0
        order_log_id = 0
        # initialize
        self.initializer()
        for student in tqdm(self.students):
            self.active_student = student
            # initialize state before student pass
            self.initializer_before_student()

            if student.id % 10 == 0:
                self.order_log[order_log_id] = [self.available_items.index(item) for item in self.items]
                order_log_id += 1

            item_order = 1
            while self.practice_terminator():
                # select item
                item = self.item_selector()
                self.active_item = item

                # solve item
                self.item_solver()

                # log activity
                if self.solve_time is not None:
                    self.log[log_id] = [student.id, item.id, item_order, student.skill, item.estimated_b, item.b,
                                        self.solve_time]
                    log_id += 1

                # update state after single response
                self.state_updater_after_item()

                item_order += 1

            # update state after student pass
            self.state_updater_after_student()

        self.log = pd.DataFrame(
            data=self.log,
            columns=['student_id', 'item_id', 'item_order', 'student_skill', 'item_difficulty', 'item_true_difficulty',
                     'solve_time'],
        ).dropna().astype({'student_id': int, 'item_id': int, 'item_order': int})
        self.order_log = pd.DataFrame(data=self.order_log,
                                      columns=[item.id for item in self.items]).dropna().astype(int)

    """
    Initializer
    """

    def initializer(self):
        # Default stuff
        self._set_seed()

        student_count = self.setting['student_count']
        item_count = self.setting['item_count']
        skill_distribution = self.setting['skill_distribution']
        b_distribution = self.setting['b_distribution']

        skills = sorted(skill_distribution(student_count, **self.setting['skill_distribution_kwargs']))
        random.shuffle(skills)
        self.students = [
            Student(id, skill)
            for id, skill in enumerate(skills)
        ]

        self._set_seed()
        sorted_difficulties = sorted(b_distribution(item_count, **self.setting['b_distribution_kwargs']))
        self.items = [
            Item(id, difficulty, self.setting['a'])
            for id, difficulty in enumerate(sorted_difficulties)
        ]
        self.true_b = [item.b for item in self.items]
        self.true_b = pd.DataFrame(data={'true_b': self.true_b}, index=range(item_count))

        self.log = np.full((student_count * item_count, 7), fill_value=np.nan)
        self.order_log = np.full((student_count, item_count), fill_value=np.nan)
        self._set_seed()

        # other extensions for different scenarios
        self._chain_executor(self.setting.get('initializer_chain', ()))

    def create_fixed_scramble(self):
        self.item_order = [i for i in range(len(self.items))]
        random.shuffle(self.item_order)

    """
    Initializer before student
    """

    def initializer_before_student(self):
        self.available_items = self.items[:]
        self._chain_executor(self.setting.get('initializer_before_student_chain', ()))

    def scramble_and_sort(self):
        random.shuffle(self.available_items)
        self.available_items = sorted(self.available_items, key=lambda item: item.estimated_b)

    def epsilon(self):
        if random.random() < self.setting['epsilon']:
            random.shuffle(self.available_items)
            self.random_students_id.append(self.active_student.id)

    def fixed_scramble(self):
        self.available_items.sort(key=lambda item: self.item_order[item.id])

    def set_remaining_time(self):
        self.remaining_solve_time = self.setting['remaining_solve_time'](self.true_b['true_b'])

    """
    Practice terminator
    """

    def practice_terminator(self):
        return self._chain_executor_cumulative(self.setting.get('practice_terminator_chain', ()))

    def available_item(self, *args):
        return len(self.available_items) > 0

    def attrition_termination(self, result):
        return result and self.remaining_solve_time > 0

    """
    Item selector
    """

    def item_selector(self):
        return self._chain_executor_cumulative(self.setting.get('item_selector_chain', ()))

    def first_item(self, *args):
        return self.available_items.pop(0)

    def random_item(self, *args):
        return self.available_items.pop(random.randint(0, len(self.available_items) - 1))

    def random_from_window(self, *args):
        window_size = min(self.setting['k'], len(self.available_items))
        return self.available_items.pop(random.randint(0, window_size - 1))

    """
    Item solver
    """

    def item_solver(self):
        self._chain_executor(self.setting.get('item_solver_chain', ()))

    def log_time_model(self):
        student = self.active_student
        item = self.active_item

        self.solve_time = item.b + item.a * student.skill + np.random.normal(0, 1)
        student.solved_items += 1

    def attrition(self):
        self.remaining_solve_time -= self.solve_time
        if self.remaining_solve_time < 0:
            self.solve_time = None

    """
    State updater after item
    """

    def state_updater_after_item(self):
        self._chain_executor(self.setting.get('state_updater_after_item_chain', ()))

    def constant_learning(self):
        # skill from -3 to 3
        max_gain = self.setting['max_gain']
        self.active_student.skill += max_gain / len(self.items)

    def step_learning(self):
        if not hasattr(self.active_student, 'has_learned') and \
                random.random() < self.setting['learn_prob']:
            max_gain = self.setting['max_gain']
            self.active_student.skill += max_gain
            self.active_student.has_learned = True

    def steep_learning(self):
        max_gain = self.setting['max_gain']
        if self.active_student.skill < max_gain:
            base = self.setting['learning_period']
            if self.active_student.solved_items < base:
                skill_increment = max_gain * log(1 / self.active_student.solved_items + 1, base)
                self.active_student.skill += skill_increment

    """
    State updater after student
    """

    def state_updater_after_student(self):
        self._chain_executor(self.setting.get('state_updater_after_student_chain', ()))

    def update_b_estimates(self):
        if (self.active_student.id + 1) % self.setting['adaptation_after'] == 0:
            self._update_b_estimates()

    def update_b_estimates_once(self):
        if (self.active_student.id + 1) == self.setting['first_k']:
            self._update_b_estimates()

    def update_b_estimates_epsilon(self):
        if self.active_student.id in self.random_students_id:
            df = pd.DataFrame(
                data=self.log,
                columns=['student_id', 'item_id', 'item_order', 'student_skill', 'item_difficulty',
                         'item_true_difficulty', 'solve_time'],
            )
            self._update_b_estimates(df[df.student_id.isin(self.random_students_id)])

    def _update_b_estimates(self, df=None):
        if df is None:
            df = pd.DataFrame(
                data=self.log,
                columns=['student_id', 'item_id', 'item_order', 'student_skill', 'item_difficulty',
                         'item_true_difficulty', 'solve_time'],
            )
        for item, estimate in zip(self.items, df.groupby('item_id')['solve_time'].mean()[0:]):
            item.estimated_b = estimate

    def get_true_difficulty_params(self):
        return self.true_b['true_b'].mean(), self.true_b['true_b'].std()

    def get_estimated_difficulty_params(self):
        estimated_difficulty_params = self.log.groupby('item_id')['solve_time'].describe()['mean']
        return estimated_difficulty_params.mean(), estimated_difficulty_params.std()


def plot_mean_times(full_log, scenarios=[], title="Estimated item difficulties", colors=[], path=None):
    assert scenarios
    if colors:
        palette = [
            sns.color_palette("Paired", 12)[color]
            for color in colors
        ]
    else:
        palette = None
    fig, ax = plt.subplots(figsize=(4.67, 3.3))
    ax = sns.lineplot(data=full_log[full_log.scenario.isin(scenarios)],
                      x='item_id',
                      y='solve_time',
                      hue='scenario',
                      palette=palette,
                      ax=ax)
    ax = sns.lineplot(data=full_log.groupby('item_id')['item_true_difficulty'].max(),
                      ax=ax, label='true difficulty', color='Y')
    ax.set_title(title)
    ax.set_xlabel('Item')
    ax.set_ylabel('Difficulty')

    if path:
        plt.savefig(path)
    else:
        plt.show()


def __sort_item_ids(group):
    return list(group[['item_id', 'item_difficulty']].sort_values(by='item_difficulty', kind='mergesort').item_id)


def plot_mean_order_convergence(full_log, scenarios=[], window_size=10, path=None, labels=None,
                                title='Item order convergence (Spearman corr. coef.)'):
    assert scenarios
    ideal_order = list(full_log.groupby('item_id')['item_true_difficulty'].max().sort_values().index)

    correlations = [
        (i // window_size + 1, spearmanr(ideal_order[:len(row)], row)[0],
         labels[scenarios.index(scenario_name)] if labels else scenario_name)
        for scenario_name in scenarios
        for i, row in enumerate(
            full_log[full_log.scenario == scenario_name].groupby('student_id').apply(lambda g: list(g.item_id)))
    ]
    df = pd.DataFrame(correlations, columns=['i', 'correlation', 'scenario'])

    fig, ax = plt.subplots(figsize=(4.67, 3.3))
    ax = sns.lineplot(data=df, x='i', y='correlation', hue='scenario', ci=None, ax=ax)
    ax.set_title(title)
    ax.set_xlabel('{}s of students'.format(window_size))
    ax.set_ylabel('Mean correlation')
    ax.set_ylim(-0.25, 1)

    if path:
        plt.savefig(path)
    else:
        plt.show()


def student_normal(n, mean=0.0, std=1.0):
    return np.random.normal(mean, std, n)


def item_normal(n, mean=50, std=0.9):
    return np.random.normal(mean, std, n)


def modify_dict(dict1, dict2):
    modified = {**dict1}
    for key, value in dict2.items():
        if key not in modified:
            modified[key] = value
        else:
            assert type(value) == list, 'Dictionaries contain contradicting keys'
            modified[key] = deepcopy(modified[key]) + value
    return modified


"""
SETTINGS
"""

"""
Initial distributions
"""

STUDENT_NORMAL = {
    'skill_distribution': student_normal,
}

ITEM_NORMAL = {
    'b_distribution': item_normal,
    'b_distribution_kwargs': {'mean': 20, 'std': 1},
}

"""
Learning modes
"""

NO_LEARNING = {
    'state_updater_after_item_chain': [],
}

CONSTANT_LEARNING = {
    'state_updater_after_item_chain': ['constant_learning'],
}

STEP_LEARNING = {
    'state_updater_after_item_chain': ['step_learning'],
    'learn_prob': 0.05,
    'max_gain': 6,
}

STEEP_LEARNING = {
    'state_updater_after_item_chain': ['steep_learning'],
    'learning_period': 10,
    'max_gain': 6,
}

"""
Student pass modes
"""

IN_ORDER_PASS = {
    'item_selector_chain': ['first_item'],
}

RANDOM_PASS = {
    'item_selector_chain': ['random_item'],
}

ADAPTATION_PASS = {
    'initializer_before_student_chain': ['scramble_and_sort'],
    'item_selector_chain': ['first_item'],
    'state_updater_after_student_chain': ['update_b_estimates'],
    'adaptation_after': 20,
}

FIRST_K_RANDOM_PASS = {
    'initializer_before_student_chain': ['scramble_and_sort'],
    'item_selector_chain': ['first_item'],
    'state_updater_after_student_chain': ['update_b_estimates_once'],
    'first_k': 20,
}

EPSILON_GREEDY_PASS = {
    'initializer_before_student_chain': ['scramble_and_sort', 'epsilon'],
    'item_selector_chain': ['first_item'],
    'state_updater_after_student_chain': ['update_b_estimates_epsilon'],
    'epsilon': 0.05,
}

"""
Other modifications
"""

ATTRITION = {
    'initializer_before_student_chain': ['set_remaining_time'],
    'item_solver_chain': ['attrition'],
    'practice_terminator_chain': ['attrition_termination'],
    'remaining_solve_time': lambda true_b: sum(true_b) * 0.6,
}

"""
Option dictionaries for easier configuration
"""

ORDERS = {
    'in order': IN_ORDER_PASS,
    'random order': RANDOM_PASS,
    'adaptation order': ADAPTATION_PASS,
    'first k random': FIRST_K_RANDOM_PASS,
    'epsilon greedy': EPSILON_GREEDY_PASS,
}

LEARNING_MODES = {
    'no': NO_LEARNING,
    'constant': CONSTANT_LEARNING,
    'step': STEP_LEARNING,
    'steep': STEEP_LEARNING,
}


def main():
    full_log = pd.DataFrame()
    config = json.load(open('scenarios.json'))

    for scenario_name in config['scenarios']:
        print("Simulating {}...".format(scenario_name))
        scenario = config['scenarios'][scenario_name]
        setting = {
            **STUDENT_NORMAL,
            **ITEM_NORMAL,
            **config['global_setting'],
            **ORDERS[scenario['order']],
            **LEARNING_MODES[scenario['learning']],
            **scenario['setting'],
        }
        if scenario.get('attrition'):
            setting = modify_dict(setting, ATTRITION)

        simulation = Simulation(setting)
        simulation.run()
        simulation.log['scenario'] = scenario_name
        full_log = full_log.append(simulation.log)

    # Store or reload simulation results for convenience
    full_log.to_pickle('data/simulation_data.pickle')
    # full_log = pd.read_pickle('data/simulation_data.pickle')

    plot_mean_times(full_log, ['incremental ordered', 'incr. small ordered'],
                    colors=[1, 11],
                    path='plots/problem-ordering-bias-a.svg')
    plot_mean_times(full_log, ['diverse ordered', 'diverse + att. ordered', 'const. + att. ordered'],
                    colors=[1, 5, 9],
                    path='plots/problem-ordering-bias-b.svg')
    plot_mean_times(full_log, ['incremental ordered', 'step ordered', 'steep ordered'],
                    colors=[1, 7, 3],
                    path='plots/problem-ordering-bias-c.svg')
    plot_mean_times(full_log, ['incremental random', 'step random', 'steep random'],
                    colors=[1, 7, 3],
                    path='plots/problem-ordering-bias-d.svg')

    plot_mean_order_convergence(full_log,
                                ['incr. small 20 random', 'incr. small 0.05 greedy', 'incr. small adaptation 20'],
                                path='plots/order-convergence-mean-correlation-a.svg')
    plot_mean_order_convergence(full_log,
                                ['incremental 20 random', 'incremental 0.05 greedy', 'incremental adaptation 20'],
                                path='plots/order-convergence-mean-correlation-b.svg')
    plot_mean_order_convergence(full_log,
                                ['diverse + att. 20 random', 'diverse + att. 0.05 greedy',
                                 'diverse + att. adaptation 20'],
                                path='plots/order-convergence-mean-correlation-c.svg')
    plot_mean_order_convergence(full_log,
                                ['const. + att. 20 random', 'const. + att. 0.05 greedy', 'const. + att. adaptation 20'],
                                path='plots/order-convergence-mean-correlation-d.svg')


if __name__ == '__main__':
    main()
