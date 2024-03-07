import unittest
import warnings
import pathlib
from GC_TLA.plopper import Arch

"""
    Classes and Tests and numbered to induce an intentional order in the Python unittest module.
    It is expected that failing a lower-ordered test MAY RESULT in failures of greater-ordered tests.

    As such, if/when tests fail, implementation should address the lowest-class, lowest-test issue first.
    After passing this lowest-order test, any other tests that continue to fail should be strictly higher unless your fix is worse than the original problem.
"""

class Test_Architecture_0_Validation(unittest.TestCase):
    """
        Test that the architecture initializes as expected with helpful exceptions and useful comparision operator
    """
    def test_arch_0_default_assign_TypeError(self):
        with self.assertRaises(TypeError):
            Arch(gpus='1')

    def test_arch_1_special_hostfile(self):
        dummy_hostfile = pathlib.Path('dummy_hostfile')
        N_NODES = 4
        with open(dummy_hostfile,'w') as f:
            f.write("\n".join([f"host_{i}" for i in range(N_NODES)]))
        arch = Arch(hostfile=dummy_hostfile)
        self.assertEqual(arch.nodes, N_NODES)
        dummy_hostfile.unlink()

    def test_arch_2_derived_MPI_ranks(self):
        dummy_hostfile = pathlib.Path('dummy_hostfile')
        N_NODES = 4
        N_GPUS = 3
        N_RANKS_PER_NODE = 20
        with open(dummy_hostfile,'w') as f:
            f.write("\n".join([f"host_{i}" for i in range(N_NODES)]))
        with self.subTest(i=0):
            arch = Arch(ranks_per_node=N_RANKS_PER_NODE, hostfile=dummy_hostfile)
            self.assertEqual(arch.mpi_ranks, N_RANKS_PER_NODE * N_NODES)
        with self.subTest(i=1):
            arch = Arch(gpus=N_GPUS, hostfile=dummy_hostfile)
            self.assertEqual(arch.mpi_ranks, N_GPUS * N_NODES)
        dummy_hostfile.unlink()

    def test_arch_3_derived_not_initialized(self):
        basic_arch = Arch()
        derived = basic_arch.get_derived()
        new_values = {}
        for attr in derived:
            arch_val = getattr(basic_arch, attr)
            if type(arch_val) is int:
                new_values[attr] = arch_val + 1
            elif type(arch_val) in (str, pathlib.Path):
                new_values[attr] = str(arch_val) + '_mutate'
            else:
                self.fail(f"Unexpected type for derived attribute {attr}: {type(arch_val)}")

        for idx, (key, value) in enumerate(new_values.items()):
            with self.subTest(i=idx):
                with self.assertRaises(ValueError):
                    new_arch_dict = {key: value}
                    Arch(**new_arch_dict)

    def test_arch_4_equality(self):
        basic_arch = Arch()
        new_values = {}
        dummy_hostfile = pathlib.Path('dummy_hostfile')
        N_NODES = 4
        # Call set_comparable to ensure the list is up-to-date, but don't let the object update state if it was erroneous
        original_comparable = [_ for _ in basic_arch.comparable]
        eq_check_comparable = set(basic_arch.set_comparable()).difference(basic_arch.get_derived())
        basic_arch.comparable = original_comparable
        for attr in eq_check_comparable:
            arch_val = getattr(basic_arch, attr)
            if type(arch_val) is int:
                new_values[attr] = arch_val + 1
            elif type(arch_val) in (str, pathlib.Path):
                new_values[attr] = str(arch_val) + "_mutate"
            elif attr == 'hostfile':
                if type(arch_val) is type(None):
                    with open(dummy_hostfile,'w') as f:
                        f.write("\n".join([f"host_{i}" for i in range(N_NODES)]))
                else:
                    with open(dummy_hostfile,'w') as f:
                        with open(arch_val, 'r') as r:
                            f.write(r.readlines())
                        f.write('\nand one more')
                new_values[attr] = dummy_hostfile
            else:
                self.fail(f"Unexpected type for {attr}: {type(arch_val)}")

        # Sub-tests for each attribute, one at a time
        idx = 0
        for idx, (key, value) in enumerate(new_values.items()):
            with self.subTest(i=idx):
                new_arch_dict = {key: value}
                # Derived key should not permit construction
                new_arch = Arch(**new_arch_dict)
                self.assertFalse(basic_arch == new_arch)
        idx = idx+1
        with self.subTest(i=idx):
            new_arch_dict = {'machine_identifier': basic_arch.machine_identifier+'_mutate'}
            new_arch = Arch(**new_arch_dict)
            # Changing machine identifier should NOT make equality fail
            self.assertTrue(basic_arch == new_arch)
        dummy_hostfile.unlink()

if __name__ == '__main__':
    unittest.main(verbosity=2)

