import matplotlib.pyplot as plt
import pandas as pd
import os

folder_path = '/AlignTDS/src/scores/'

# Get a list of all files in the folder
files = os.listdir(folder_path)

coords = {
    "med_instruct_full": [[1000, 10000, 20000, 52000], [0,0,0,0]],
    "med_instruct_lora": [[1000, 10000, 20000, 52000], [0,0,0,0]],
    "alpaca_full": [[1000, 10000, 25000, 52000], [0,0,0,0]],
    "alpaca_lora": [[1000, 10000, 25000, 52000], [0,0,0,0]]
}

for llm in ["llama-2-7b"]:
    csv_files = [file for file in files if file.endswith('.csv') and file.startswith(llm)]
    for method in ["KL", "TP", "TR"]:
        for csv in csv_files:
            df = pd.read_csv(f"/AlignTDS/src/scores/{csv}", header=0)
            pre_str = "full" if "full" in csv else "lora"
            if "alpaca" in csv:
                if "10000" in csv:
                    coords["alpaca_" + pre_str][1][1] = df[method][0]
                elif "1000" in csv:
                    coords["alpaca_" + pre_str][1][0] = df[method][0]
                elif "25000" in csv:
                    coords["alpaca_" + pre_str][1][2] = df[method][0]
                else:
                    coords["alpaca_" + pre_str][1][3] = df[method][0]
            elif "MedInstruct" in csv:
                if "10000" in csv:
                    coords["med_instruct_" + pre_str][1][1] = df[method][0]
                elif "1000" in csv:
                    coords["med_instruct_" + pre_str][1][0] = df[method][0]
                elif "20000" in csv:
                    coords["med_instruct_" + pre_str][1][2] = df[method][0]
                else:
                    coords["med_instruct_" + pre_str][1][3] = df[method][0]
        
        plt.clf()

        for k, v in coords.items():
            plt.plot(v[0], v[1], label=f"{k}")
            # plt.plot(v[0], [0.78*v for v in v[1]], label=f"{k}")
            # plt.plot(v[0], [0.60*v for v in v[1]], label=f"{k}")

        plt.xlabel('Scale')
        plt.ylabel('Scores')
        plt.title(method)

        # plt.legend()

        plt.savefig(f'{method}.png')


# # Data for the first line   
# x1 = [1000, 10000, 25000, 52000]
# y1 = [0.6795, 0.7910, 0.8018, 0.78]

# # Data for the second line
# x2 = [1000, 10000, 25000, 52000]
# y2 = [0.14038, 0.1992, 0.2115, 0.2277]

# # Data for the first line
# x3 = [1000, 10000, 25000, 52000]
# y3 = [0.5325, 0.6175, 0.7814, 0.7636]

# # Data for the second line
# x4 = [1000, 10000, 25000, 52000]
# y4 = [0.1341 , 0.1364, 0.1342, 0.1455]

# # Plotting the first line
# plt.plot(x1, y1, label='llama-2-7b-full-MedInstruct')

# # Plotting the second line
# plt.plot(x2, y2, label='llama-2-7b-lora-MedInstruct')

# # Plotting the third line
# plt.plot(x3, y3, label='llama-2-7b-full-alpaca')

# # Plotting the fourth line
# plt.plot(x4, y4, label='llama-2-7b-lora-alpaca')

# # Adding labels and title
# plt.xlabel('Scale')
# plt.ylabel('Scores')
# plt.title('KL Divergence')

# # Adding a legend
# plt.legend()

# # Display the plot
# plt.show()
# plt.savefig('sample_plot.png')