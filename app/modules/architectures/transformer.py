from .base_architecture import ArchitectureModule

class TransformerArchitecture(ArchitectureModule):
    key          = "transformer"
    label        = "Transformer"
    accent_color = "#39d353"
    diagram_type = "transformer"
    trainable    = True
    description  = (
        "<h3>Transformer</h3>"
        "<code>Tokens → Embed+PosEnc → [MultiHeadAttn + FFN] × L → Output</code><br>"
        "Self-attention lets every token attend to every other token in parallel. "
        "No recurrence — fully parallelisable. "
        "Foundation of GPT, BERT, T5 and virtually all modern AI.<br><b>Live training enabled.</b>"
    )
