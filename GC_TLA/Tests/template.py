import unittest
import warnings
import pathlib

"""
    Classes and Tests and numbered to induce an intentional order in the Python unittest module.
    It is expected that failing a lower-ordered test MAY RESULT in failures of greater-ordered tests.

    As such, if/when tests fail, implementation should address the lowest-class, lowest-test issue first.
    After passing this lowest-order test, any other tests that continue to fail should be strictly higher unless your fix is worse than the original problem.
"""
class Test_CLASS_0_REASON(unittest.TestCase):
    """
        Test that ...
    """
    def test_CLASS_0_UNIT(self):
        pass

if __name__ == '__main__':
    unittest.main(verbosity=2)

