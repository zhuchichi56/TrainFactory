import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.size': 12,
    'font.weight': 'bold'
})

colors = ['lightcoral', 'lightsalmon', 'lightgreen']

models = ['FANNO(LLaMA2) VS. Alpaca-52k', #[666, 21, 118]
          'FANNO(LLaMA2) VS. WizardLM-70k', #[437, 7, 361]
          'FANNO(LLaMA2) VS. LLaMA2', #[758, 1, 46]
          'FANNO(LLaMA2) VS. Alpaca-GPT4', #[654, 16, 135]
          'FANNO(LLaMA2) VS. Alpaca-Cleaned', #[588, 20, 197]
          'FANNO(LLaMA2) VS. Muffin', #[684, 3, 118]
          'FANNO(LLaMA2) VS. OSS-Instruct', #[661, 16, 128]
          'FANNO(LLaMA2) VS. Genie', #[701, 3, 101]
          'FANNO(LLaMA2) VS. Humback $\mathregular{M_{0}^{*}}$', # [520, 12, 273]
          'FANNO(LLaMA2) VS. LIMA' # [637, 10, 157]
            ]

total_sum = 805
win_values = [666,
              437,
              758,
              654,
              588,
              684,
              661,
              701,
              520,
              637]

tie_values = [21, 
              7, 
              1,
              16,
              20,
              3,
              16,
              3,
              12,
              10]

lose_values = [118,
               361,
               46,
               135,
               197,
               118,
               128,
               101,
               273,
               157]

# Sort models based on win_values
models = [x for _, x in sorted(zip(win_values, models), reverse=True)]

# Sort tie_values based on win_values
tie_values = [x for _, x in sorted(zip(win_values, tie_values), reverse=True)]

# Sort lose_values based on win_values
lose_values = [x for _, x in sorted(zip(win_values, lose_values), reverse=True)]

# Sort win_values in descending order
win_values = sorted(win_values, reverse=True)

x = np.arange(len(models))
width = 0.5

fig, ax = plt.subplots(figsize=(12, 6))

bottom_values = np.zeros(len(models))

ax.bar(x, win_values, width, label='Our Wins', color=colors[0])
bottom_values += np.array(win_values)

ax.bar(x, tie_values, width, label='Tie', bottom=bottom_values, color=colors[1])
bottom_values += np.array(tie_values)

ax.bar(x, lose_values, width, label='Opponent Wins', bottom=bottom_values, color=colors[2])

ax.set_ylabel('Alpaca Eval', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=20, ha='right')
ax.legend()

for i in range(len(models)):
    ax.text(x[i], win_values[i] - 50, str(win_values[i]), ha='center', va='bottom')
    ax.text(x[i], win_values[i] + tie_values[i] + 5, str(tie_values[i]), ha='center', va='bottom')
    ax.text(x[i], win_values[i] + tie_values[i] + lose_values[i] + 5, str(lose_values[i]), ha='center', va='bottom')

for val in range(100, total_sum, 100):
    ax.axhline(y=val, color='gray', linestyle='--', linewidth=0.5, zorder=0)

ax.axhline(y=total_sum / 2, color='blue', linestyle='--', linewidth=1,zorder=0)
ax.text(-0.4, total_sum / 2 + 10, '402', color='blue', ha='right', va='center')

# sixth_set_index = 10
# ax.axvline(x=sixth_set_index + 0.5, color='orange', linestyle='-.', linewidth=2)

plt.tight_layout()
plt.savefig('/home/admin/AlpacaEval3.svg')
