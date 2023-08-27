#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

create_placeholders = __import__('0-create_placeholders').create_placeholders


x, y = create_placeholders(1, 1)
assert(x.name == 'x:0')
assert(y.name == 'y:0')
