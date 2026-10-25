from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError


class PasswordService:
    """Passwords are stored only as Argon2 hashes: slow, salted, one-way scrambles.

    Even if the database leaks, the original passwords cannot be read back out of it.
    """

    def __init__(self, **hasher_options):
        # Defaults are the library's safe ones; tests pass cheap settings to run fast.
        self._hasher = PasswordHasher(**hasher_options)
        # A hash of nothing in particular, used to make "no such user" take as long as
        # "wrong password" (see AuthService.login).
        self._dummy_hash = self._hasher.hash("not-a-real-password")

    def hash(self, password: str) -> str:
        return self._hasher.hash(password)

    def verify(self, password_hash: str, password: str) -> bool:
        try:
            return self._hasher.verify(password_hash, password)
        except (VerificationError, InvalidHashError):
            return False

    def verify_dummy(self, password: str) -> None:
        self.verify(self._dummy_hash, password)
