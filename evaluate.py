# evaluate.py — Test accuracy, confusion matrix, classification report

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from config import CLASS_NAMES


def evaluate_model( model, test_loader, device ) -> dict:
    model.eval()
    model.to( device )
    all_preds, all_labels = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to( device ), labels.to( device )
            preds   = model( images ).argmax( dim=1 )
            correct += ( preds == labels ).sum().item()
            total   += labels.size( 0 )
            all_preds.extend( preds.cpu().numpy() )
            all_labels.extend( labels.cpu().numpy() )

    return {
        "accuracy":         correct / total,
        "confusion_matrix": confusion_matrix( all_labels, all_preds, labels=[0, 1, 2] ),
        "report":           classification_report( all_labels, all_preds,
                                target_names=CLASS_NAMES, output_dict=True, zero_division=0 ),
        "all_preds":        np.array( all_preds ),
        "all_labels":       np.array( all_labels ),
    }