from benchmark.scenarios.benign_internal import (
    create_benign_internal_scenario,
)
from benchmark.scenarios.direct_leakage import (
    create_direct_leakage_scenario,
)
from benchmark.scenarios.summary_leakage import (
    create_summary_leakage_scenario,
)

from benchmark.scenarios.memory_mediated import (
    create_memory_mediated_scenario,
)


def get_initial_scenarios():
    return [
        create_direct_leakage_scenario(),
        create_summary_leakage_scenario(),
        create_benign_internal_scenario(),
        create_memory_mediated_scenario(),
    ]