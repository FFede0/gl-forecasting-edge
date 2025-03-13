import matplotlib.colors as mcolors
from carbontracker import parser
import matplotlib.pyplot as plt
from parse import parse
import pandas as pd
import os


def plot_network_metrics(
    avg: pd.DataFrame, std: pd.DataFrame, dir: str, network: str, key: str
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
      axs[idx].set_xlabel(key, fontsize = 14)
    axs[idx].set_ylabel(col.replace("_", " "), fontsize = 14)
  plt.savefig(
    os.path.join(dir, f"{network}_metrics.png"),
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

colors = list(mcolors.TABLEAU_COLORS.values())

for event, event_logs in logs.groupby("event"):
  event_logs = event_logs.drop("event", axis = "columns")
  key = "epoch"
  if not "fit" in event:
    key = "call"
    event_logs.rename(
      columns = {"epoch": key, "epoch_durations (s)": f"{key}_durations (s)"}, 
      inplace = True
    )
  plot_dir = os.path.join(gossip_dir, f"plot_{event}")
  os.makedirs(plot_dir, exist_ok = True)
  # average by node
  summary_data = pd.DataFrame()
  for network, network_logs in event_logs.groupby("network"):
    avg = network_logs.groupby(key).mean(numeric_only = True)
    std = network_logs.groupby(key).std(numeric_only = True)
    plot_network_metrics(avg, std, plot_dir, network, key)
    summary_data = pd.concat(
      [summary_data, avg.reset_index()], ignore_index = True
    )
  # average averages over network
  avg = summary_data.groupby(key).mean(numeric_only = True)
  std = summary_data.groupby(key).std(numeric_only = True)
  plot_network_metrics(avg, std, plot_dir, "average", key)

