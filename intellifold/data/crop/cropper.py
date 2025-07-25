# Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro
#
# Licensed under the MIT License:
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from intellifold.data.types import Tokenized


class Cropper(ABC):
    """Abstract base class for cropper."""

    @abstractmethod
    def crop(
        self,
        data: Tokenized,
        max_tokens: int,
        random: np.random.RandomState,
        max_atoms: Optional[int] = None,
        chain_id: Optional[int] = None,
        interface_id: Optional[int] = None,
    ) -> Tokenized:
        """Crop the data to a maximum number of tokens.

        Parameters
        ----------
        data : Tokenized
            The tokenized data.
        max_tokens : int
            The maximum number of tokens to crop.
        random : np.random.RandomState
            The random state for reproducibility.
        max_atoms : Optional[int]
            The maximum number of atoms to consider.
        chain_id : Optional[int]
            The chain ID to crop.
        interface_id : Optional[int]
            The interface ID to crop.

        Returns
        -------
        Tokenized
            The cropped data.

        """
        raise NotImplementedError
