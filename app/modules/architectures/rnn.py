from .base_architecture import ArchitectureModule

class RNNArchitecture(ArchitectureModule):
    key          = "rnn"
    label        = "RNN / LSTM"
    accent_color = "#39d353"
    diagram_type = "rnn"
    trainable    = True
    description  = (
        "<h3>Recurrent / LSTM Network</h3>"
        "<code>x₁ → [h₁] → [h₂] → … → [hT] → Output</code><br>"
        "Hidden state carries memory across timesteps. "
        "LSTM gates (forget / input / output) solve the vanishing gradient problem "
        "of vanilla RNNs. Best for sequences, time series, language.<br><b>Live training enabled.</b>"
    )
