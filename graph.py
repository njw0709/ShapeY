import sys
import hydra
from hydra.core.config_store import ConfigStore
import logging
from shapey.utils.configs import ShapeYConfig

sys.path.append("./step4_graph_results")
from graph_exclusion_top1error_v2 import graph_exclusion_top1

log = logging.getLogger(__name__)
cs = ConfigStore.instance()
cs.store(name="defaultconf", node=ShapeYConfig, group="grp")


@hydra.main(config_path="./conf", config_name="config")
def graph(cfg: ShapeYConfig) -> None:
    log.info("graph exclusion top1 error...")
    step4_completed = graph_exclusion_top1(cfg)
    if step4_completed:
        log.info("done graph exclusion top1 error.")
    else:
        log.error("failed to graph exclusion top1 error.")
        return
    log.info("done with {}!".format(cfg.network.name))


if __name__ == "__main__":
    graph()
