import re
import warnings

"""
    * find should be a list of regexes each with at least one group () to match
        + Note that regexes are taken as-is.
        + To make escaping easier, for prefixes and suffixes around the matching group, it is best to utilize the prefix/suffix find-replace tuples
    * prefix|suffix should be a list of tuples of equal length, where each tuple has 2 strings
        + As of now, only static strings are supported (not regexes)
        + The first string is matched and removed from the original input
        + The second string is replaced in the new output where the match was removed
"""
# Default list of one tuple matching nothing to nothing (used as default padding for __init__())
empty_from_to = [("","",),]

class FindReplaceRegex:
    def __init__(self, find, prefix=None, suffix=None, expectPrefixMatch=False, expectSuffixMatch=False):
        if type(find) is str:
            find = (find,)
            # It is reasonable to only pass one prefix/suffix in this case
            if prefix is not None and type(prefix[0]) is not tuple:
                prefix = (prefix,)
            if suffix is not None and type(suffix[0]) is not tuple:
                suffix = (suffix,)
        # Reasonable warnings to remove some hard edges on class utilization
        for idx, entry in enumerate(find):
            if '(' not in entry or ')' not in entry:
                warnings.warn(f"Find regex #{idx}, '{entry}', might not contain a regex capture (), which will fail to find-replace", UserWarning)
            if prefix is not None:
                if len(prefix) <= idx:
                    raise ValueError(f"No prefix for find-regex #{idx}: {entry}. Each find must have a prefix if prefixes are defined!")
                else:
                    if type(prefix[idx]) is not tuple or len(prefix[idx]) != 2:
                        raise ValueError(f"Prefix {idx}, '{prefix[idx]}', must be a tuple of two values")
                    if entry.startswith(prefix[idx][0]) and not expectPrefixMatch:
                        warnings.warn(f"Find regex #{idx}, '{entry}', begins with associated prefix, '{prefix[idx]}'. Matching strings will have to take the form '{prefix[idx]}{entry}', which may not meet your expectation (pass expectPrefixMatch=True to supress)", UserWarning)
            if suffix is not None:
                if len(suffix) <= idx:
                    raise ValueError(f"No suffix for find-regex #{idx}: {entry}. Each find must have a suffix if suffixes are defined!")
                else:
                    if type(suffix[idx]) is not tuple or len(suffix[idx]) != 2:
                        raise ValueError(f"Suffix {idx}, '{suffix[idx]}', must be a tuple of two values")
                    if entry.startswith(suffix[idx][0]) and not expectSuffixMatch:
                        warnings.warn(f"Find regex #{idx}, '{entry}', ends with associated suffix, '{suffix[idx]}'. Matching strings will have to take the form '{suffix[idx]}{entry}', which may not meet your expectation (pass expectSuffixMatch=True to supress)", UserWarning)
        self.find = find
        self.nitems = len(self.find)
        REQUIRED_ELEMS = 2*self.nitems

        # Repeated code for each of these attributes
        for attrName, passedValue in zip(['prefix', 'suffix'], [prefix, suffix]):
            # Be nice about wrapping/replacing default values
            if passedValue is None:
                passedValue = empty_from_to * self.nitems
            elif type(passedValue) is tuple and type(passedValue[0]) is str:
                passedValue = [passedValue,]
            # Validation of required length for each find-regex
            nAttrItems = sum(map(len, passedValue))
            if nAttrItems != REQUIRED_ELEMS:
                raise ValueError(f"{attrName} must have 2-element tuple per element in the find regex list (got {nAttrItems}, needed {REQUIRED_ELEMS})")
            else:
                setattr(self, attrName, passedValue)

        # Magic variables that can try to predict common use patterns and ease function paramaterization
        self.iter_idx = None
        self.invert_direction = 0

    def __str__(self):
        return str({'find': self.find,
                    'prefix': self.prefix,
                    'suffix': self.suffix,
                    'iter_idx': self.iter_idx,
                    'invert_direction': self.invert_direction})

    def __iter__(self):
        # Enumeration just to set up the magic variable
        for idx, regex in enumerate(self.find):
            self.iter_idx = idx
            yield regex

    def replace(self, match, to, string):
        # Automatically handle the expected replacement patterns
        if to is None or to == "":
            return re.sub(self.wrap(match, noInvert=True), "", string)
        else:
            return re.sub(self.wrap(match), self.wrap(to), string)

    def wrap(self, wrap, direction=None, idx=None, noInvert=False):
        # When direction|idx are None, attempt to use magic variables to predict correct output
        # Actual values passed to direction may be ['from'==0,'to'==1]
        # Actual values passed to idx may be in the range of values in self.find
        if direction is None:
            direction = self.invert_direction
        if type(direction) is str:
            if direction.lower() == 'from':
                direction = 0
            elif direction.lower() == 'to':
                direction = 1
        if direction not in [0, 1]:
            raise ValueError(f"Could not parse direction '{direction}', must be in ['from', 'to'] or [0, 1]")
        if idx is None:
            if self.iter_idx is None:
                if self.nitems == 1:
                    # Only case where these can both be None and we unambiguously match user expectations
                    idx = 0
                else:
                    raise ValueError(f"Index to wrap is poorly defined! Please define an index")
            else:
                idx = self.iter_idx
        # Magic updates for next call to match (usually expect opposite direction)
        if not noInvert:
            self.invert_direction = int(not direction)
        return self.prefix[idx][direction] + wrap + self.suffix[idx][direction]

    def findReplace(self, string, substitution, lookup_match_substitution=None):
        """
            This is the main method you should call on instances of this class
            Take a line of text to be updated (string)
            and the replacement for the regex-matched portion of the find element (substitution)
            then return the modified string

            If replacements should be dynamic, use a dictionary-like (lookup_match_substitution) that
            accepts a key of the regex-matched portion and returns your desired substitution for that text
        """
        for search_for_str in self:
            already_handled = set()
            for re_match in re.finditer(search_for_str, string):
                match = re_match.group()
                if match in already_handled:
                    continue
                already_handled.add(match)
                if lookup_match_substitution is not None:
                    substitution = lookup_match_substitution[match]
                string = self.replace(match, substitution, string)
        return string

