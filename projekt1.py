import random
import numpy as np

dice_faces = [4, 6, 8, 12, 20]

def create_dice_dist(n):
    return [1/n] * n

probability_mass = [create_dice_dist(n) for n in dice_faces]

convolution_sum = probability_mass[0]
for i in probability_mass[1:]:
    convolution_sum = np.convolve(convolution_sum, i)

sum_less_than_10 = convolution_sum[0:6]
sum_bigger_than_45 = convolution_sum[45-5:50-5+1]
print(sum_less_than_10)
print(sum_bigger_than_45)

win_probability = sum_less_than_10.sum() + sum_bigger_than_45.sum()
print(win_probability)


# Monte Carlo
n_trials = 1000

def generate_trials(start, end, n_trials):
    return [random.randint(start, end) for x in range(n_trials)]


def generate_probability_for_trials(n):
    trial_results = np.array([generate_trials(1, i, n) for i in dice_faces]).sum(axis=0)

    mc_sum_under_10 = [i for i in trial_results if i <= 10]
    mc_sum_over_45 = [i for i in trial_results if i >= 45]
    number_of_wins = len(mc_sum_under_10) + len(mc_sum_over_45)
    probability_of_wins = number_of_wins/n
    return probability_of_wins

print(generate_probability_for_trials(n_trials))

number_of_trials = [1000, 3000, 5000, 8000, 10000, 20000, 50000, 80000, 100000]
relative_error = []
p = [generate_probability_for_trials(n) for n in number_of_trials]
print(p)


for n in number_of_trials:
    relative_error_per_trial_set = []
    for i in range(0, 30):
        relative_error_per_trial_set.append(abs(generate_probability_for_trials(n)-win_probability)/win_probability)
    relative_error.append(sum(relative_error_per_trial_set)/len(relative_error_per_trial_set))

print(relative_error)