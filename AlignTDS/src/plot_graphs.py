import matplotlib.pyplot as plt
import pickle
import sys
import pandas as pd

curves = []

pair = sys.argv[1]

with open(f"..//AlignTDS/saved_logits/curves/{pair}_curves.pkl", "rb") as f:
    curves = pickle.load(f)

x = []
y = {}
step_score = {}
label = list(curves[0].keys()).remove("step")

score = {"KL": [], "TP": [], "TR": []}

print(f"Total tokens : {len(curves)}")
for point in curves:
    x.append(point["step"])
    # step_score[point["step"]] 
    for method in ["KL", "TP", "TR"]:
        score[method].append(point[method])
# print(y)

# print(f"Total tokens : {len(score)}")
for k,v in score.items():
    score[k] = [sum(score[k])/len(score[k])]
    print(f"Value for method : {k} and score : {score[k]}")

df = pd.DataFrame.from_dict(score)
df.to_csv(f"..//AlignTDS/src/scores/{pair}.csv", index=False)
print(x)
plt.scatter(x, score["KL"], label='Sample Data', color='blue', marker='x')
plt.xlabel('Position')
plt.ylabel('Score')
plt.title('Sample Graph')
plt.legend()
plt.xlim(0, max(x))

plt.savefig('sample_plot.png')