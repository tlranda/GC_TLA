from GC_TLA.utils import Configurable as C

i1 = C()
C._configure(a=1)
i2 = C()
C._remove_configure('a')
i3 = C()
i4 = C()
i4._update_from_core(b=2)
i5 = C()
i6 = C()
i6._update_from_core(c=3)
i7 = C()
i7._remove_from_core('c')
i8 = C()

tests = {'blank': i1,
         'with_a_from_classmethod': i2,
         'without_a_from_classmethod': i3,
         'with_b_from_instmethod': i4,
         'also_gets_b_from_previous_instmethod': i5,
         'with_c_from_instmethod': i6,
         'without_c_from_instmethod': i7,
         'end_state': i8}

for (reason, instance) in tests.items():
    print(reason, str(instance))

