import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Task 1 ---
dice_faces = [4, 6, 8, 12, 20]

# Generate the probability mass for each face of a dice as an array
def create_dice_dist(n):
    return [1/n] * n

probability_mass = [create_dice_dist(n) for n in dice_faces]

# Calculate the probability for different sums by convolving the probability mass
# for 2 dices at a time, then taking the result of that to convolve
# with the probability mass of the next dice
convolution_sum = probability_mass[0]
for i in probability_mass[1:]:
    convolution_sum = np.convolve(convolution_sum, i)


# Create a table
min_sum = len(dice_faces)          # 5
max_sum = sum(dice_faces)          # 50
s_values = np.arange(min_sum, max_sum + 1)

table = pd.DataFrame({
    's': s_values,
    'P(S=s)': convolution_sum
})

print("--- Task 1 ---")
print(table)
print("------------------\n\n")


# --- Task 2 ---
# Splice the array to get the sections that satisfies the winning condition
sum_less_than_10 = convolution_sum[0 : 10 - min_sum + 1]
sum_bigger_than_45 = convolution_sum[45 - min_sum : max_sum - min_sum + 1]

# Sum all the winning probabilities to get the total win probability
win_probability = sum_less_than_10.sum() + sum_bigger_than_45.sum()
print("--- Task 2 ---")
print(f"Win probability = {win_probability} ---> {win_probability * 100}%")
print("------------------\n\n")


# --- Task 3 ---
# Monte Carlo
n_trials = 1000

# Simulate 1000 trials for a dice where param 'n_faces' is the number of faces
def generate_trials(n_faces, n_trials):
    return [random.randint(1, n_faces) for x in range(n_trials)]

# Return the probability of wins for n amount of trials
def generate_probability_for_trials(n_trials):
    trial_results = np.array([generate_trials(i, n_trials) for i in dice_faces]).sum(axis=0)

    mc_sum_under_10 = [i for i in trial_results if i <= 10]
    mc_sum_over_45 = [i for i in trial_results if i >= 45]
    number_of_wins = len(mc_sum_under_10) + len(mc_sum_over_45)
    probability_of_wins = number_of_wins/n_trials
    return probability_of_wins

win_probability_1000_trials = generate_probability_for_trials(n_trials)

print("--- Task 3 ---")
print(f"Win probability for 1000 trials = {win_probability_1000_trials} ---> {win_probability_1000_trials*100} %")
print("------------------\n\n")


# --- Task 4 ---
number_of_trials = [1000, 3000, 5000, 8000, 10000, 20000, 50000, 80000, 100000]
win_probability_for_trials = []
all_repeats = []

# For each trial set (1000, 3000, 5000, ...), we repeat the simulation for 20 times
# to reduce sampling noice due to the simulation's random nature. The final probability
# for that trial set is then averaged across all repeated simulations within that set.
for n in number_of_trials:
    probability_per_trial_set = [generate_probability_for_trials(n) for i in range(20)]
    all_repeats.append(probability_per_trial_set)
    average_probability = sum(probability_per_trial_set) / len(probability_per_trial_set)
    win_probability_for_trials.append(average_probability)

# Plot results
print("Plotting results for Task 4...")
plt.figure(figsize=(8, 5))
plt.plot(number_of_trials, win_probability_for_trials, marker='o', label='Simulated estimate')
plt.axhline(win_probability, color='red', linestyle='--', label='Exact value (Task 2)')

plt.xscale('log')
plt.xlabel('Number of trials (log scale)')
plt.ylabel('Estimated P(win)')
plt.title('Monte Carlo estimate of P(win) vs. number of trials')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.savefig('task4_convergence.png', dpi=150)
plt.show(block=False)


# --- Task 5 ---
relative_error = []

# For every trial set (1000, 3000, 5000, ...), we calculate the relative error for all repeated simulations
# within that set and take the average of all the values as the final relative error 
# for that trial set
for probability_per_trial_set in all_repeats:
    errors = [abs(p - win_probability) / win_probability for p in probability_per_trial_set]
    relative_error.append(sum(errors) / len(errors))

# Plot results
print("Plotting results for Task 5...")
plt.figure(figsize=(8, 5))
plt.plot(number_of_trials, relative_error, marker='o', label='Average relative error')
plt.axhline(0.10, color='red', linestyle='--', label='10% threshold')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Number of trials (log scale)')
plt.ylabel('Average relative error (log scale)')
plt.title('Relative error vs. number of trials')
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.savefig('task5_relative_error.png', dpi=150)
plt.show()