from __future__ import annotations

from dataclasses import field, dataclass

from gossiplearning.models import MetricName, MetricValue, NodeId, Time


@dataclass
class MessageHistoryLog:
    from_node: int
    to_node: int
    time_sent: int
    time_received: int


@dataclass
class UpdateHistoryLog:
    node: int
    from_time: int
    to_time: int


NodeMetricsHistory = dict[MetricName, list[MetricValue]]
MetricsHistory = dict[NodeId, NodeMetricsHistory]


@dataclass
class History:
    stopped_time: dict[NodeId, Time] = field(default_factory=dict)
    messages: list[MessageHistoryLog] = field(default_factory=list)
    trainings: list[UpdateHistoryLog] = field(default_factory=list)
    nodes_training_history: MetricsHistory = field(default_factory=dict)
    nodes_test_history: MetricsHistory = field(default_factory=dict)
