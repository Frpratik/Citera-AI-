from fastapi import APIRouter, Depends

from app.api.deps import CurrentUser, ServicesDep, limit_by_address
from app.api.schemas import LoginRequest, RegisterRequest, TokenOut, UserOut

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=201, dependencies=[Depends(limit_by_address)])
def register(body: RegisterRequest, services: ServicesDep):
    return UserOut.model_validate(services.auth.register(body.email, body.password))


@router.post("/login", response_model=TokenOut, dependencies=[Depends(limit_by_address)])
def login(body: LoginRequest, services: ServicesDep):
    return TokenOut(access_token=services.auth.login(body.email, body.password))


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser):
    return UserOut.model_validate(user)
