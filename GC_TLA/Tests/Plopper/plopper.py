import unittest
import warnings
import pathlib
from GC_TLA.Plopper.plopper import Plopper

"""
    Classes and Tests and numbered to induce an intentional order in the Python unittest module.
    It is expected that failing a lower-ordered test MAY RESULT in failures of greater-ordered tests.

    As such, if/when tests fail, implementation should address the lowest-class, lowest-test issue first.
    After passing this lowest-order test, any other tests that continue to fail should be strictly higher unless your fix is worse than the original problem.
"""
class Test_Plopper_0_Constructor(unittest.TestCase):
    """
        Test that ...
    """
    def test_plopper_0_(self):
        pass

    # To be implemented:
    # Test proper construction
    # Test that template filling does NOT create any files if findReplace is None and force_write remains False (default)
    # Test that template filling DOES create an identical file copy if findReplace is None and force_write is True
    # Test that created file from template filling has permissions 755
    # Test that use_raw_template does not clobber an existing template via templateExecute
    # Test that templateExecute's random destination is properly created as a random filename
    # Test that ALL commands in buildTemplateCmds have to succeed or else executors return UnableToExecute infinity-value
    # Test that valid buildExecutorCmds that can operate on filled-in template produce a correct result via templateExecute

if __name__ == '__main__':
    unittest.main(verbosity=2)

