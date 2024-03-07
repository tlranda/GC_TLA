import unittest
import warnings
from GC_TLA.utils import FindReplaceRegex

"""
    Classes and Tests and numbered to induce an intentional order in the Python unittest module.
    It is expected that failing a lower-ordered test MAY RESULT in failures of greater-ordered tests.

    As such, if/when tests fail, implementation should address the lowest-class, lowest-test issue first.
    After passing this lowest-order test, any other tests that continue to fail should be strictly higher unless your fix is worse than the original problem.
"""

class Test_FRR_0_Constructors(unittest.TestCase):
    """
        Test for warnings and ValueErrors in the __init__()
        Passing ALL of these should indicate that well-formed constructors will:
            * Only raise warnings when appropriate
            * Permit normal operation without internal error
    """
    def test_const_0_warn_on_bad_find(self):
        # When a find regex lacks a capture group (), we should be warned during construction
        with self.assertWarns(UserWarning):
            FindReplaceRegex((r"A[0-9]+",))

    def test_const_1_warn_dup_prefix(self):
        # When a prefix duplicates the start of a find regex, we should be warned during construction IF the user did not indicate they knew that was the case
        with self.assertWarns(UserWarning):
            FindReplaceRegex((r"Z(A[0-9]+)",),
                             prefix=[("Z","")])

    def test_const_2_warn_dup_suffix(self):
        # When a suffix duplicates the end of a find regex, we should be warned during construction IF the user did not indicate they knew that was the case
        with self.assertWarns(UserWarning):
            FindReplaceRegex((r"(A[0-9]+)X",),
                             suffix=[("X","")])

    def test_const_3_raise_insufficient_prefix(self):
        # When less prefixes are defined than find regexes, a ValueError should be raised
        with self.assertRaises(ValueError):
            FindReplaceRegex((r"sk(i)p", r"Z(A[0-9]+)X",),
                             prefix=[("just","one"),])

    def test_const_4_raise_insufficient_suffix(self):
        # When less suffixes are defined than regexes, a ValueError should be raised
        with self.assertRaises(ValueError):
            FindReplaceRegex((r"sk(i)p", r"Z(A[0-9]+)X",),
                             suffix=[("just","one"),])

class Test_FRR_1_Basic_Functionality(unittest.TestCase):
    """
        Covers most practical use cases for the class via findReplace()
        Passing ALL of these should indicate that non-edge cases meet expectations
    """
    def setUp(self):
        # Single point to define a simple valid instance that is used in test cases
        self.FRR_instance = FindReplaceRegex((r"(A[0-9]+)",),
                                             prefix=[("Z","X",)],
                                             suffix=[("X","Z",)])

    def test_basic_func_0_unaltered(self):
        # When none of the capture regexes are present, there should be no mutation in the string
        test_prompt = "This sample should be unaltered"
        test_replace = "REPLACEMENT"
        test_correct = test_prompt
        test_result = self.FRR_instance.findReplace(test_prompt, test_replace)
        self.assertEqual(test_result, test_correct)

    def test_basic_func_1_one_substitution(self):
        # Basic matching should preserve the prefix, suffix, and replace match with replacement
        test_prompt = "This sample has ZA1X 1 substitution"
        test_replace = "REPLACEMENT"
        test_correct = "This sample has XREPLACEMENTZ 1 substitution"
        test_result = self.FRR_instance.findReplace(test_prompt, test_replace)
        self.assertEqual(test_result, test_correct)

    def test_basic_func_2_empty_substitution(self):
        # Empty substitution DESTROYS the prefix, match, and suffix entirely from the string
        test_prompt = "This sample has ZA1X 1 substitution"
        test_replace = None
        test_correct = "This sample has  1 substitution"
        test_result = self.FRR_instance.findReplace(test_prompt, test_replace)
        self.assertEqual(test_result, test_correct)

    def test_basic_func_3_multi_substitution(self):
        # Multiple matches on one line should be handled in a single call
        test_prompt = "ZA093XThis sample has 2 substitutions ZA81437813X"
        test_replace = "REPLACEMENT"
        test_correct = "XREPLACEMENTZThis sample has 2 substitutions XREPLACEMENTZ"
        test_result = self.FRR_instance.findReplace(test_prompt, test_replace)
        self.assertEqual(test_result, test_correct)

    def test_basic_func_4_dynamic_substitution(self):
        # Dynamic lookups for matches are supported, meaning the same regex can match many behaviors
        test_prompt = "ZA093XThis sample has 2 substitutions ZA81437813X"
        test_replace = "REPLACEMENT"
        test_lookup = {"A093": "Match1", "A81437813": "Match2"}
        test_correct = "XMatch1ZThis sample has 2 substitutions XMatch2Z"
        test_result = self.FRR_instance.findReplace(test_prompt, test_replace, test_lookup)
        self.assertEqual(test_result, test_correct)

class Test_FRR_2_DupPrefSuf_Functionality(unittest.TestCase):
    """
        This is something of an edge-case, but prefixes and suffixes are not considered in the find
        regex, only in the resulting substitutions. This can lead to strange behaviors, thus the
        constructor checks, but it should be permissible to use them to these effects when desired

        Passing ALL of these tests indicates that duplicate prefixes/suffixes will be required for matching
    """
    def setUp(self):
        # Simple valid instance to match test cases
        self.FRR_instance = FindReplaceRegex((r"Z(A[0-9]+)X",),
                                             prefix=[("Z","X",)],
                                             suffix=[("X","Z",)],
                                             expectPrefixMatch=True,
                                             expectSuffixMatch=True)

    def test_func_0_empty_replace_dup_prefix_suffix_OK(self):
        # When the replacement is None or "", prefix and suffix are NOT re-applied
        test_prompt = "ZZA0XX is a match"
        test_replace = ""
        test_correct = " is a match"
        test_result = self.FRR_instance.findReplace(test_prompt, test_replace)
        self.assertEqual(test_result, test_correct)

    def test_func_1_nonempty_dup_prefix_suffix_OK(self):
        # When replacement is NOT None or "", prefix and Suffix ARE re-applied
        test_prompt = "ZZA0XX is a match"
        test_replace = "the match"
        test_correct = "Xthe matchZ is a match"
        test_result = self.FRR_instance.findReplace(test_prompt, test_replace)
        self.assertEqual(test_result, test_correct)

    def test_func_2_nonempty_dup_prefix_suffix_unchanged(self):
        # When you don't surround a regex-find substring with your prefix/suffix, the string is unchanged due to incomplete match
        test_prompt = "ZA0X is a match"
        test_replace = "the match"
        test_incorrect = "Xthe matchZ is a match"
        test_correct = test_prompt
        test_result = self.FRR_instance.findReplace(test_prompt, test_replace)
        self.assertNotEqual(test_result, test_incorrect)
        self.assertEqual(test_result, test_correct)

    def test_func_3_nonempty_dup_prefix_unchanged(self):
        # When you surround a regex-find substring with your prefix but NOT your suffix, the string is unchanged due to incomplete match
        test_prompt = "ZZA0X is a match"
        test_replace = "the match"
        test_incorrect = "Xthe matchZ is a match"
        test_correct = test_prompt
        test_result = self.FRR_instance.findReplace(test_prompt, test_replace)
        self.assertNotEqual(test_result, test_incorrect)
        self.assertEqual(test_result, test_correct)

    def test_func_4_nonempty_dup_suffix_unchanged(self):
        # When you surround a regex-find substring with your suffix but NOT your prefix, the string is unchanged due to incomplete match
        test_prompt = "ZA0XX is a match"
        test_replace = "the match"
        test_incorrect = "Xthe matchZ is a match"
        test_correct = test_prompt
        test_result = self.FRR_instance.findReplace(test_prompt, test_replace)
        self.assertNotEqual(test_result, test_incorrect)
        self.assertEqual(test_result, test_correct)

# Not creating a suite as we're not sophisticated enough to need that yet
if __name__ == '__main__':
    unittest.main(verbosity=2)

