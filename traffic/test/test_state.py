from traffic.detector import state_from_class


def test_state_from_model_class():
    assert state_from_class('vehicular_red') == 'RED'
    assert state_from_class('vehicular_yellow') == 'YELLOW'
    assert state_from_class('vehicular_green') == 'GREEN'
    assert state_from_class('vehicular_red_and_green_arrow') == 'LEFT_ARROW'
    assert state_from_class('vehicular_etc') == 'UNKNOWN'
    assert state_from_class('vehicular_green_arrow(down)') == 'UNKNOWN'
