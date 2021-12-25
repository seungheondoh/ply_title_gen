from typing import List, Callable, Union, Dict

class Vocab:
    """Vocab class"""
    def __init__(
        self,
        list_of_tokens: List[str] = None,
        padding_token: str = "<pad>",
        unknown_token: str = "<unk>",
        sos_token: str = "<sos>",
        eos_token: str = "<eos>",
        reserved_tokens: List[str] = None,
        token_to_idx: Dict[str, int] = None,
    ):
        self._unknown_token = unknown_token
        self._padding_token = padding_token
        self._sos_token = sos_token
        self._eos_token = eos_token
        self._reserved_tokens = reserved_tokens
        self._special_tokens = []

        for tkn in [
            self._padding_token,
            self._sos_token,
            self._eos_token,
            self._unknown_token,
        ]:
            if tkn:
                self._special_tokens.append(tkn)

        if self._reserved_tokens:
            self._special_tokens.extend(self._reserved_tokens)

        if list_of_tokens:
            self._special_tokens.extend(
                list(
                    filter(lambda elm: elm not in self._special_tokens, list_of_tokens)
                )
            )

        self._token_to_idx, self._idx_to_token = self._build(self._special_tokens)

        if token_to_idx:
            self._sort_index_according_to_user_specification(token_to_idx)

        self._embedding = {}

    def to_indices(self, tokens: Union[str, List[str]]) -> Union[int, List[int]]:
        """Looks up indices of text tokens according to the vocabulary
        Args:
            tokens (Union[str, List[str]]): a source token or tokens to be converted
        Returns:
            Union[int, List[int]]: a token index or a list of token indices according to the vocabulary
        """
        if isinstance(tokens, list):
            return [
                self._token_to_idx[tkn]
                if tkn in self._token_to_idx
                else self._token_to_idx[self._unknown_token]
                for tkn in tokens
            ]
        else:
            return (
                self._token_to_idx[tokens]
                if tokens in self._token_to_idx
                else self._token_to_idx[self._unknown_token]
            )

    def to_tokens(self, indices: Union[int, List[int]]) -> Union[str, List[str]]:
        """Converts token indices to tokens according to the vocabulary
        Args:
            indices (Union[int, List[int]]): a source token index or token indices to be converted
        Returns:
            Union[str, List[str]]: a token or a list of tokens according to the vocabulary
        """
        if isinstance(indices, list):
            return [self._idx_to_token[idx] for idx in indices]
        else:
            return self._idx_to_token[indices]

    def _build(self, list_of_tokens):
        token_to_idx = {tkn: idx for idx, tkn in enumerate(list_of_tokens)}
        idx_to_token = list_of_tokens
        return token_to_idx, idx_to_token

    def _sort_index_according_to_user_specification(self, token_to_idx):
        # Sanity checks
        if not set(token_to_idx.keys()).issubset(self._token_to_idx.keys()):
            raise ValueError(
                "User-specified token_to_idx mapping can only contain "
                "tokens that will be part of the vocabulary."
            )
        if len(set(token_to_idx.values())) != len(token_to_idx):
            raise ValueError("User-specified indices must not contain duplicates.")
        if min(token_to_idx.values()) < 0 or max(token_to_idx.values()) >= len(
            self._token_to_idx
        ):
            raise ValueError(
                "User-specified indices must not be < 0 or >= the number of tokens "
                "that will be in the vocabulary. The current vocab contains {}"
                "tokens.".format(len(self._token_to_idx))
            )

        # Update index ordering
        for token, new_idx in token_to_idx.items():
            old_idx = self._token_to_idx[token]
            ousted_token = self._idx_to_token[new_idx]

            self._token_to_idx[token] = new_idx
            self._token_to_idx[ousted_token] = old_idx
            self._idx_to_token[old_idx] = ousted_token
            self._idx_to_token[new_idx] = token

    def __len__(self):
        return len(self._token_to_idx)

    @property
    def token_to_idx(self):
        return self._token_to_idx

    @property
    def idx_to_token(self):
        return self._idx_to_token

    @property
    def padding_token(self):
        return self._padding_token

    @property
    def unknown_token(self):
        return self._unknown_token

    @property
    def sos_token(self):
        return self._sos_token

    @property
    def eos_token(self):
        return self._eos_token

    @property
    def embedding(self):
        return self._embedding

    @embedding.setter
    def embedding(self, array):
        self._embedding = array