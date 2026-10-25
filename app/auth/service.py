from app.auth.passwords import PasswordService
from app.auth.tokens import TokenService
from app.core.errors import AuthenticationError
from app.repositories.users import UserRecord, UserRepository

# One message for "no such email" and "wrong password", so nobody can use the login
# form to find out which emails have accounts.
INVALID_LOGIN_MESSAGE = "Invalid email or password."


class AuthService:
    def __init__(self, users: UserRepository, passwords: PasswordService, tokens: TokenService):
        self._users = users
        self._passwords = passwords
        self._tokens = tokens

    def register(self, email: str, password: str) -> UserRecord:
        return self._users.create(email, self._passwords.hash(password))

    def login(self, email: str, password: str) -> str:
        found = self._users.get_credentials(email)
        if found is None:
            self._passwords.verify_dummy(password)  # same time as a real check
            raise AuthenticationError(INVALID_LOGIN_MESSAGE)
        user, password_hash = found
        if not self._passwords.verify(password_hash, password):
            raise AuthenticationError(INVALID_LOGIN_MESSAGE)
        return self._tokens.issue(user.id)

    def user_from_token(self, token: str) -> UserRecord:
        user = self._users.get(self._tokens.verify(token))
        if user is None:  # a valid signature for a user that no longer exists
            raise AuthenticationError("Invalid or expired token.")
        return user
