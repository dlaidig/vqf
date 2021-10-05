# SPDX-FileCopyrightText: 2021 Daniel Laidig <laidig@control.tu-berlin.de>
#
# SPDX-License-Identifier: MIT

import atexit
import os
from abc import ABC, abstractmethod

import numpy as np
import pytest
import vqf


def pytest_addoption(parser):
    parser.addoption('--nomatlab', action='store_true', default=False,
                     help='disable tests of Matlab implementation via Matlab and transplant')
    parser.addoption('--nooctave', action='store_true', default=False,
                     help='disable tests of Matlab implementation via Octave and oct2py')


class AbstractMatlabWrapper(ABC):
    """
    This class (and MatlabVQF/OctaveVQF) provides a wrapper for executing the Matlab implementation in Matlab and
    Octave via transplant and oct2py. This is only meant for unit testing purposes and contains a number of workarounds
    for limitations of transplant and oct2py (e.g. when calling static class methods and working with objects).
    Do not use in production.
    """
    is_matlab = True

    nargoutMap = {}

    def __init__(self, instance):
        self.m = instance
        self.m.addpath(os.path.join(os.path.dirname(__file__), '../vqf/matlab'))

    def __getattr__(self, item):
        # static class method
        nargout = self.nargoutMap.get(item, 1)

        def wrapper(*args):
            out = self.callStaticMethod(item, args, nargout)
            return self.flattenOutputs(out)
        return wrapper

    def __call__(self, *args, **kwargs):
        # call constructor
        obj = self.createObject(args, kwargs)

        class Wrapper:
            def __init__(self, parent):
                self.parent = parent

            def __getattr__(self, item):
                # class method
                nargout = self.parent.nargoutMap.get(item, 1)

                def wrapper(*args):
                    out = self.parent.callMethod(obj, item, args, nargout)
                    return self.parent.flattenOutputs(out)
                return wrapper

        return Wrapper(self)

    @abstractmethod
    def callStaticMethod(self, name, args, nargout):
        pass

    @abstractmethod
    def createObject(self, args, kwargs):
        pass

    @abstractmethod
    def callMethod(self, obj, name, args, nargout):
        pass

    @staticmethod
    def flattenOutputs(out):
        if isinstance(out, np.ndarray):
            return np.squeeze(out)
        elif isinstance(out, list):
            return [AbstractMatlabWrapper.flattenOutputs(v) for v in out]
        elif isinstance(out, dict):
            return {k: AbstractMatlabWrapper.flattenOutputs(v) for k, v in out.items()}
        return out


class MatlabVQF(AbstractMatlabWrapper):
    def __init__(self):
        import transplant
        super().__init__(transplant.Matlab())

        # _del_proxy sometimes seems to hang
        # quick and dirty hack: prevent deletion of the object instances by keeping a reference in this list...
        self.instances = []

    def callStaticMethod(self, name, args, nargout):
        return self.m.eval(f'@(varargin) VQF.{name}(varargin{{:}})')(*args, nargout=nargout)

    def createObject(self, args, kwargs):
        # import transplant
        # obj = self.m.VQF(*args, transplant.MatlabStruct(kwargs))
        obj = self.m.VQF()  # temporary for dummy
        self.instances.append(obj)
        return obj

    def callMethod(self, obj, name, args, nargout):
        return getattr(obj, name)(*args, nargout=nargout)


class OctaveVQF(AbstractMatlabWrapper):
    def __init__(self):
        import oct2py
        super().__init__(oct2py.Oct2Py())
        self.next = 0

    def callStaticMethod(self, name, args, nargout):
        self.m.push('tmp', tuple(args))  # tuples are converted to cell arrays
        names = [f'tmp{i + 1}' for i in range(nargout)]
        self.m.eval(f'[{",".join(names)}]=VQF.{name}(tmp{{:}});')
        return self.m.pull(names)

    def createObject(self, args, kwargs):
        name = f'obj{self.next}'
        # self.m.push('tmp', tuple(args) + tuple([kwargs]))  # tuples are converted to cell arrays
        # self.m.eval(f'{name}=VQF(tmp{{:}});')
        self.m.eval(f'{name}=VQF();')  # temporary for dummy
        self.next += 1
        return name

    def callMethod(self, obj, name, args, nargout):
        self.m.push('tmp', tuple(args))  # tuples are converted to cell arrays
        if nargout == 0:
            self.m.eval(f'{obj}.{name}(tmp{{:}});')
        else:
            names = [f'tmp{i + 1}' for i in range(nargout)]
            self.m.eval(f'[{",".join(names)}]={obj}.{name}(tmp{{:}});')
            return self.m.pull(names)


# it seems like the session fixture is called multiple times
# use global instances for now to mitigate this
_matlabInstance = None
_octaveInstance = None


def _exitMatlab():
    if _matlabInstance is not None:
        _matlabInstance.m.exit()
    if _octaveInstance is not None:
        _octaveInstance.m.exit()


atexit.register(_exitMatlab)


@pytest.fixture(scope='session')
def cls(request):
    if request.param == 'VQF':
        return vqf.VQF
    elif request.param == 'PyVQF':
        return vqf.PyVQF
    elif request.param == 'MatlabVQF':
        if request.config.getoption("--nomatlab"):
            return None
        else:
            global _matlabInstance
            if _matlabInstance is None:
                _matlabInstance = MatlabVQF()
            return _matlabInstance
    elif request.param == 'OctaveVQF':
        if request.config.getoption("--nooctave"):
            return None
        else:
            global _octaveInstance
            if _octaveInstance is None:
                _octaveInstance = OctaveVQF()
            return _octaveInstance
    else:
        raise RuntimeError(f'invalid param "{request.param}"')
