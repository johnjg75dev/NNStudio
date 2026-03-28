from .. import db

class LayerDefinition(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Metadata
    name = db.Column(db.String(50), nullable=False) # slug
    label = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    
    # Logic Reference
    type = db.Column(db.String(50), nullable=False, default='dense')
    
    # Default Config
    default_activation = db.Column(db.String(50), default='tanh')
    default_neurons = db.Column(db.Integer, default=4)

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "type": self.type,
            "default_activation": self.default_activation,
            "default_neurons": self.default_neurons
        }
