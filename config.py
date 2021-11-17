EPOCHS = 250
BATCH_SIZE = 128

TYPE = 'lstm'
REC_SIZE = 1
UNITS = None
HID_SIZE = 20

OPTIM = 'rmsprop'
LR = 0.001
ALPHA = 0.99 if OPTIM == 'rmsprop' else None
MOMENTUM = 0.5 if OPTIM == 'rmsprop' else None
BETAS = (0.9, 0.999) if OPTIM == 'adam' else None
WEIGHT_DECAY = 0