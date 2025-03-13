import matplotlib.colors as mcolors
from carbontracker import parser
import matplotlib.pyplot as plt
from parse import parse
import pandas as pd
import os


def plot_network_metrics(
    avg: pd.DataFrame, std: pd.DataFrame, gossip_dir: str, network: str
  ):
  fig, axs = plt.subplots(
    nrows = len(avg.columns), 
    sharex = True,
    figsize = (10, 4*len(avg.columns))
  )
  fig.tight_layout()
  for idx, col in enumerate(avg.columns):
    avg[col].plot(
      grid = True,
      fontsize = 14,
      ax = axs[idx],
      legend = False,
      color = colors[idx]
    )
    axs[idx].fill_between(
      x = avg.index,
      y1 = [max(y,0) for y in avg[col] - std[col]],
      y2 = avg[col] + std[col],
      color = colors[idx],
      alpha = 0.4
    )
    if idx == len(avg.columns) - 1:
      axs[idx].set_xlabel("epoch", fontsize = 14)
    axs[idx].set_ylabel(col.replace("_", " "), fontsize = 14)
  plt.savefig(
    os.path.join(gossip_dir, f"{network}_metrics.png"),
    dpi = 300,
    format = "png",
    bbox_inches = "tight"
  )
  plt.close()


gossip_dir = "../experiments/azurefunctions-dataset2019/10n_3k_15min/seed2000/gossip-MergeStrategy.AGE_WEIGHTED"
logs = pd.DataFrame()
for network in os.listdir(gossip_dir):
  log_dir = os.path.join(gossip_dir, network, "tracker")
  if os.path.exists(log_dir):
    # parse all logs
    all_logs_list = parser.parse_all_logs(
      log_dir = log_dir
    )
    # loop over logs
    for idx, log_dict in enumerate(all_logs_list):
      output_filename = log_dict["output_filename"]
      standard_filename = log_dict["standard_filename"]
      # get identifier
      node, event0, event1, _ = parse(
        "node_{}_{}_{}_{}_carbontracker_output.log", os.path.split(output_filename)[1]
      )
      # create dataframe with detailed CPU consumption
      log = log_dict["components"]
      if log["cpu"]["avg_power_usages (W)"] is not None:
        log = pd.DataFrame({
          k: v.flatten().tolist() for k, v in log["cpu"].items() if k != "devices"
        })
        # # add aggregate real and predicted consumption
        # if log_dict["pred"] is not None:
        #   for k, v in log_dict["pred"].items():
        #     if k != "equivalents":
        #       log[f"pred {k}"] = [v] * len(log)
        #     else:
        #       k2 = list(v.keys())[0]
        #       v2 = list(v.values())[0]
        #       log[f"pred {k2}"] = [v2] * len(log)
        # if log_dict["actual"] is not None:
        #   for k, v in log_dict["actual"].items():
        #     if k != "equivalents":
        #       log[f"actual {k}"] = [v] * len(log)
        #     else:
        #       k2 = list(v.keys())[0]
        #       v2 = list(v.values())[0]
        #       log[f"actual {k2}"] = [v2] * len(log)
        # add node, event, index and network information
        log["node"] = [node] * len(log)
        log["event"] = [f"{event0}_{event1}"] * len(log)
        log["epoch"] = range(1, len(log)+1)
        log["network"] = [network] * len(log)
        # concatenate
        logs = pd.concat([logs, log], ignore_index = True)

logs.to_csv(os.path.join(gossip_dir, "carbontracker.csv"), index = False)

model_fit = logs[logs["event"] == "model_fit"].drop("event", axis = "columns")
colors = list(mcolors.TABLEAU_COLORS.values())

summary_data = pd.DataFrame()
for network, network_logs in model_fit.groupby("network"):
  avg = network_logs.groupby("epoch").mean(numeric_only = True)
  std = network_logs.groupby("epoch").std(numeric_only = True)
  plot_network_metrics(avg, std, gossip_dir, network)
  summary_data = pd.concat(
    [summary_data, avg.reset_index()], ignore_index = True
  )

avg = summary_data.groupby("epoch").mean(numeric_only = True)
std = summary_data.groupby("epoch").std(numeric_only = True)
plot_network_metrics(avg, std, gossip_dir, "average")
  


for (network, node), log in model_fit.groupby(["network", "node"]):
  s = pd.DataFrame(
    log.drop(["node", "network"], axis = "columns").sum(),
    columns = ["sum"]
  )
  a = log.drop(["node", "event", "idx", "network"], axis = "columns").mean()

model_fit[
  (
    model_fit["network"] == "9"
  ) & (
    model_fit["node"] == "9"
  )
]["epoch_durations (s)"].sum()

log["avg_power_usages (W)"].sum()/1000 * 54/3600

log["actual energy (kWh)"].sum()




