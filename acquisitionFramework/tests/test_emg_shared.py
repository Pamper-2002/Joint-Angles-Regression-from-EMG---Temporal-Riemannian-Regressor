import EMG


def test_emg_uses_only_an_independent_shared_subscription(monkeypatch):
    events = []

    class Board:
        enable_dev_board_logger = staticmethod(lambda: None)
        get_exg_channels = staticmethod(lambda board: list(range(8)))

        def __init__(self, board_id, params):
            assert board_id == -2 and params.master_board == 0
            assert params.serial_number.startswith('subscriber-')
            events.append('construct')

        def prepare_session(self): events.append('prepare')
        def start_stream(self): events.append('start')
        def get_board_id(self): return 0
        def stop_stream(self): events.append('stop')
        def release_session(self): events.append('release')

    monkeypatch.setattr(EMG, 'BoardShim', Board)
    monkeypatch.setattr(EMG, 'ensure_service', lambda: events.append('ensure'))
    source = EMG.MindRoveEMG()
    source.stop()
    source.stop()
    assert events == ['ensure', 'construct', 'prepare', 'start', 'stop', 'release']
