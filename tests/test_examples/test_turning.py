def test_td_elgohary(snapshot):
    from examples.turning._01_td_elgohary import turns, turns_global

    snapshot.assert_match(turns, "turns_per_gs")
    snapshot.assert_match(turns_global, "turns_per_gs_global_frame")
