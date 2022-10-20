import sys
from ai.algorithms.gan_create_number import NumberGenerator

ng = NumberGenerator()
ng.fit(sys.argv[1])


# > python input_parameter.py 1 3 aaa