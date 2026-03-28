from .. import db

class ArchitectureDefinition(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Metadata
    name = db.Column(db.String(50), nullable=False) # slug
    label = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=True)
    accent_color = db.Column(db.String(10), default='#58a6ff')
    diagram_type = db.Column(db.String(50), default='generic')
    
    # Capabilities
    trainable = db.Column(db.Boolean, default=False)
    is_autoencoder = db.Column(db.Boolean, default=False)

    def to_dict(self):
        return {
            "id": self.id,
            "key": self.name,
            "label": self.label,
            "description": self.description,
            "category": "architectures",
            "trainable": self.trainable,
            "accent_color": self.accent_color,
            "diagram_type": self.diagram_type,
            "is_autoencoder": self.is_autoencoder,
        }
