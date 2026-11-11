from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import User
from app.routers.common import skills_by_theme, user_out
from app.schemas import RegisterIn, TokenOut, UserOut
from app.security import create_access_token, get_current_user, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register", response_model=TokenOut, status_code=status.HTTP_201_CREATED)
def register(body: RegisterIn, db: Session = Depends(get_db)) -> TokenOut:
    clash = db.scalar(
        select(User).where(or_(User.username.ilike(body.username), User.email.ilike(body.email)))
    )
    if clash:
        raise HTTPException(status.HTTP_409_CONFLICT, "Username or email already registered")
    user = User(username=body.username, email=body.email.lower(), password_hash=hash_password(body.password))
    db.add(user)
    db.flush()
    skills_by_theme(db, user.id)  # seed the six theme ratings
    db.commit()
    return TokenOut(access_token=create_access_token(user.id))


@router.post("/login", response_model=TokenOut)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)) -> TokenOut:
    ident = form.username.strip()
    user = db.scalar(select(User).where(or_(User.username.ilike(ident), User.email == ident.lower())))
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect username or password")
    return TokenOut(access_token=create_access_token(user.id))


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> UserOut:
    return user_out(user)
