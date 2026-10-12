from mindrove.board_shim import BoardShim
from semg_shared.mindrove.service import STREAM_BOARD, ensure_service, subscriber_params
from mindrove.data_filter import DataFilter, DetrendOperations

EMG_SFREQ = 500
EMG_N_CHANNEL = 8

class MindRoveEMG:
    def __init__(self):
        BoardShim.enable_dev_board_logger()
        ensure_service()
        self.board_shim = BoardShim(STREAM_BOARD, subscriber_params())
        self.board_shim.prepare_session()
        self.board_shim.start_stream()
        board_id = self.board_shim.get_board_id()
        self.exg_channels = BoardShim.get_exg_channels(board_id)
        self._stopped = False

    def getEMG(self):
        data = self.board_shim.get_board_data()
        try:
            if len(data[0]) >= 1:
                for count, channel in enumerate(self.exg_channels):
                    DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
        except Exception as e:
            print("could not detrend data. Continue to run")

        return data[:8]

    def clearBuffer(self):
        self.board_shim.get_board_data()

    def stop(self):
        """Release the streaming session exactly once."""
        if getattr(self, "_stopped", False):
            return
        self._stopped = True
        try:
            self.board_shim.stop_stream()
        except Exception:
            pass
        try:
            self.board_shim.release_session()
        except Exception:
            pass
