
def tests():
    """
    >>> tests()
    FAIL WHALE
    """
    print "Testing"
    1/0


if __name__ == "__main__":
    import doctest
    doctest.testmod()
