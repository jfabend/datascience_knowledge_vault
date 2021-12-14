Also known as Fuzzy String searching

The problem of approximate string matching is typically divided into two sub-problems: finding approximate substring matches inside a given string and finding dictionary strings that match the pattern approximately

The closeness of a match is measured in terms of the number of primitive operations necessary to convert the string into an exact match. This number is called the edit distance between the string and the pattern. The usual primitive operations are:[1]

insertion: cot → coat
deletion: coat → cot
substitution: coat → cost
These three operations may be generalized as forms of substitution by adding a NULL character (here symbolized by *) wherever a character has been deleted or inserted:

insertion: co*t → coat
deletion: coat → co*t
substitution: coat → cost