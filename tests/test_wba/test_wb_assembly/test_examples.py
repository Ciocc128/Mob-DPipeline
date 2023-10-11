import pandas as pd
from pandas._testing import assert_frame_equal

from gaitlink.wba._wb_assembly import WBAssembly
from gaitlink.wba._wb_criteria import MaxBreakCriteria, NStridesCriteria
from tests.test_wba.conftest import window


def test_simple_single_wb():
    """Simple single WB test.

    Scenario:
        - Break at the beginning gait, break
        - No left right foot

    Rules:
        - BreakCriterium
        - MinStride Inclusion Rule

    """
    wb_start_time = 5
    n_strides = 10
    strides = pd.DataFrame.from_records(
        [window(wb_start_time + i, wb_start_time + i + 1) for i in range(n_strides)]
    ).set_index("s_id")
    rules = [("break", MaxBreakCriteria(3)), ("n_strides", NStridesCriteria(4))]

    wba = WBAssembly(rules)
    wba.assemble(strides)

    assert len(wba.wbs_) == 1
    single_wb = next(iter(wba.wbs_.values()))
    assert len(single_wb) == n_strides
    assert_frame_equal(single_wb, strides)

    assert len(wba.excluded_stride_list_) == 0

    # assert len(wba.excluded_wb_list_) == 0
    # assert len(wba.excluded_stride_list_) == 0
    # assert len(wba.exclusion_reasons_) == 0
    # assert len(wba.stride_exclusion_reasons_) == 0


def test_simple_break_center():
    """Test gait sequence with a break in the center.

    Scenario:
        - long gait sequence with a break in the center
        - no left right

    Rules:
        - BreakCriterium
        - MinStride Inclusion Rule

    Outcome:
        - Two WBs (one for one after the break)
        - no strides discarded
    """
    wb_start_time = 5
    n_strides = 20
    strides = pd.DataFrame.from_records(
        [window(wb_start_time + i, wb_start_time + i + 1) for i in range(n_strides)]
    ).set_index("s_id")
    strides = strides.drop(strides.index[7:11])
    rules = [("break", MaxBreakCriteria(3)), ("n_strides", NStridesCriteria(4))]

    wba = WBAssembly(rules)
    wba.assemble(strides)

    # assert len(wba.excluded_wb_list_) == 0
    assert len(wba.excluded_stride_list_) == 0
    # assert len(wba.exclusion_reasons_) == 0
    # assert len(wba.stride_exclusion_reasons_) == 0

    assert len(wba.wbs_) == 2
    wbs = list(wba.wbs_.values())
    assert wbs[0].iloc[0]["start"] == wb_start_time
    assert wbs[0].iloc[-1]["end"] == wb_start_time + 7
    assert_frame_equal(wbs[0], strides.iloc[:7])

    assert wbs[1].iloc[0]["start"] == 16
    assert wbs[1].iloc[-1]["end"] == wb_start_time + n_strides
    assert_frame_equal(wbs[1], strides.iloc[7:])


# TODO: Add a couple more simple test cases
