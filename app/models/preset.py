from .. import db

class Preset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Metadata
    label = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    
    # Network Configuration
    arch_key = db.Column(db.String(50), nullable=False)
    func_key = db.Column(db.String(50), nullable=False)
    hidden_layers = db.Column(db.Integer, default=1)
    neurons = db.Column(db.Integer, default=4)
    activation = db.Column(db.String(50), default='tanh')
    optimizer = db.Column(db.String(50), default='adam')
    loss = db.Column(db.String(50), default='bce')
    lr = db.Column(db.Float, default=0.01)
    dropout = db.Column(db.Float, default=0.0)
    weight_decay = db.Column(db.Float, default=0.0)

    def to_dict(self):
        return {
            "id": self.id,
            "key": f"db_preset_{self.id}",
            "label": self.label,
            "description": self.description,
            "arch_key": self.arch_key,
            "func_key": self.func_key,
            "hidden_layers": self.hidden_layers,
            "neurons": self.neurons,
            "activation": self.activation,
            "optimizer": self.optimizer,
            "loss": self.loss,
            "lr": self.lr,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "custom": True  # Flag for frontend to show delete button
        }
