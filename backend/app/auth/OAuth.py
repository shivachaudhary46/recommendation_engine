from fastapi.security import OAuth2PasswordBearer
from fastapi import Depends, status, HTTPException
from sqlmodel import select
import jwt
from jwt.exceptions import InvalidTokenError
import os
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv
load_dotenv()
from typing import Union
from typing import Annotated

from app.vector_db.database import SessionDep
from app.models.models import User
from app.models.schemas import TokenData
from app.config import settings

oauth_scheme2 = OAuth2PasswordBearer(tokenUrl="/token")

def get_user(username: str, session: SessionDep) -> User:
    statement = select(User).where(User.username == username)
    results = session.exec(statement)
    account = results.first()

    if not account:
        return None

    return account

def authenticate_user(session: SessionDep, username: str, password: str) :
    user = get_user(username, session)

    from pwdlib import PasswordHash
    hasher = PasswordHash.recommended()

    if not user:
        return None
    if not isinstance(password, str):
        return none
    if not hasher.verify(password, user.hashed_password):
        return None 
    return user

def create_access_token(data: dict, expire_time: Union[timedelta, None] = None):
    to_encode = data.copy()
    
    if expire_time:
        expire = datetime.now(timezone.utc) + expire_time
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    
    # Add validation
    if not settings.SECRET_KEY:
        raise ValueError("SECRET_KEY is not configured")

    encoded_jwt = jwt.encode(
        to_encode, 
        str(settings.SECRET_KEY),  # Ensure it's a string
        algorithm=settings.ALGO
    )
    return encoded_jwt

async def get_current_user(token: Annotated[str, Depends(oauth_scheme2)], session: SessionDep):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGO])
        username = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except InvalidTokenError:
        raise credentials_exception
    user = get_user(username=token_data.username, session=session)
    if user is None:
        raise credentials_exception
    return user

def role_required(allowed_roles: list):
    def wrapper(current_user: User = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        return current_user
    return wrapper