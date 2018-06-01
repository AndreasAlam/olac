# general
import os
import sys


def set_path(level=1, change_path=True):
    """Set the path to the olac/olac directory or insert it in the path.

    Parameters:
    -----------
    level : int
            The number of levels you are removed from the olac parent
            directory. Maximum of two levels are implemented.

    change_path : boolean
                  True - Change the working directory
                  False - Inset the path to the python path

    Returns:
    --------
    None : nonetype
           The function does not return anything
    """
    if level == 1:
        base = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    elif level == 2:
        base = os.path.normpath(
            os.path.normpath(os.getcwd() + os.sep + os.pardir)
            + os.sep + os.pardir + '/olac'
        )
    else:
        raise Exception('Level {0} is not a valid value'.format(level))
    if change_path:
        os.chdir(base)
    else:
        sys.path.insert(0, base)
    return(None)
