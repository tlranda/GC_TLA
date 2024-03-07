import unittest
import warnings
import pathlib
from GC_TLA.plopper import MetricIDs, Executor

"""
    Classes and Tests and numbered to induce an intentional order in the Python unittest module.
    It is expected that failing a lower-ordered test MAY RESULT in failures of greater-ordered tests.

    As such, if/when tests fail, implementation should address the lowest-class, lowest-test issue first.
    After passing this lowest-order test, any other tests that continue to fail should be strictly higher unless your fix is worse than the original problem.
"""

class Test_MIDs_0_Validation(unittest.TestCase):
    """
        Test that the class function for MetricIDs enum validates when expected and fails to validate when expected
    """
    def test_mids_0_validate_NotOK_Only(self):
        self.assertTrue(MetricIDs.validate_infinity_mapping({MetricIDs.NotOK: None}))

    def test_mids_1_validate_all_but_NotOK(self):
        self.assertTrue(MetricIDs.validate_infinity_mapping({MetricIDs.TimeOut: None,
                                                             MetricIDs.BadReturnCode: None,
                                                             MetricIDs.BadParse: None,
                                                             MetricIDs.UnableToExecute: None}))
    def test_mids_2_invalidate_partial_spec(self):
        self.assertFalse(MetricIDs.validate_infinity_mapping({MetricIDs.TimeOut: None}))

    def test_mids_3_invalidate_bogus_spec(self):
        self.assertFalse(MetricIDs.validate_infinity_mapping({'a':'b'}))

    # Eventually this validator may warn about unused keys, but for now that is not implemented and not tested

class Test_Executor_0_Validation(unittest.TestCase):
    """
        Use very basic test to validate that code executes, is parsed, and summarized
    """
    def test_executor_0_complete_execution(self):
        exe = Executor()
        basic_path = pathlib.Path('executor_test_case')
        self.assertEqual(exe.execute(basic_path, lambda x, y: ["echo '1'"]), 1)
        # Clean up resulting test data
        for i in range(3):
            new_path = basic_path.with_name(f"{basic_path.name}_{i}.log")
            if new_path.exists():
                new_path.unlink()

    # To be added:
    # .cleanup() subclassed with an error -- strict_cleanup should produce Exception, without should be Warning
    # Any of the below indented should return most specific available error, so check it AND check that NotOK is default
    # for valid mappings that don't include the specific reason for that test
    #       Set ignore_runtime_failure and arbitrarily fail execution (Should result in BadParse?)
    #       Don't set ignore_runtime_failure and arbitrarily fail execution (Should result in BadReturnCode)
    #       Commands that arbitrarily time out and don't recover (Should result in TimeOut)

# Not creating a suite as we're not sophisticated enough to need that yet
if __name__ == '__main__':
    unittest.main(verbosity=2)

