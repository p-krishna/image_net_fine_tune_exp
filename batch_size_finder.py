# batch_size_finder.py
# Finds the largest batch size that fits in RAM without crashing
import torch
import torchvision.models as models
import torch.nn as nn
from time import time

model = models.vgg16( weights=None )   # no download needed
model.classifier[6] = nn.Linear( 4096, 3 )
model.train()
criterion = nn.CrossEntropyLoss()

for bs in [32, 64, 128, 256]:
# for bs in [256, 512, 1024]:
    try:
        x = torch.randn( bs, 3, 224, 224 )
        y = torch.randint( 0, 3, ( bs, ) )
        start_time = time()
        out  = model( x )
        loss = criterion( out, y )
        loss.backward()
        print( f"  batch_size={bs:>4}  ✓  OK  ( {time() - start_time:.2f} seconds )" )
    except MemoryError as e:
        print( f"  batch_size={bs:>4}  ✗  OOM: {e}" )
        break
    except RuntimeError as e:
        print( f"  batch_size={bs:>4}  ✗  RuntimeError: {e}" )
    except Exception as e:
        print( f"  batch_size={bs:>4}  ✗  UnexpectedError: {e}" )