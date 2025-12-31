from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Annotated, Optional

router = APIRouter(
    prefix="/users",
    tags=["users"]
)

from sqlmodel import Session, select, desc
from app.models.models import User, Movie, Reaction, Interaction, Watchlist
from app.vector_db.database import SessionDep
from app.models.schemas import UserCreate, UserResponse
from app.auth.OAuth import get_current_user, authenticate_user
from pwdlib import PasswordHash 

hasher = PasswordHash.recommended()

def get_user_by_username(session: Session, username: str) -> Optional[User]:
    statement = select(User).where(User.username == username)
    return session.exec(statement).first()

def get_all_users(session: Session, skip: int = 0, limit: int = 100) -> List[User]:
    statement = select(User).offset(skip).limit(limit)
    users = session.exec(statement).all()
    return users  

def delete_user_by_id(session: Session, user_id: int): 
    reactions = session.exec(
        select(Reaction).where(Reaction.user_id == user_id)
    ).all() 
    watchlists = session.exec(
        select(Watchlist).where(Watchlist.user_id == user_id)
    ).all() 
    interactions = session.exec(
        select(Interaction).where(Interaction.user_id == user_id)
    )
    for react in reactions: 
        session.delete(react)
    
    for watchlist in watchlists:
        session.delete(watchlist)
        
    for interact in interactions: 
        session.delete(interact)

    user = session.get(User, user_id)
    if not user: 
        return False
    
    session.delete(user)
    session.commit() 
    return True

# Create User
@router.post("/", response_model=UserResponse)
def create_new_user(user_data: UserCreate, session: SessionDep):
    """Create a new user"""
    try:
        existing = get_user_by_username(session, user_data.username)
        if existing:
            raise HTTPException(status_code=400, detail="Username already exists")

        user = User(
            username=user_data.username,
            email=user_data.email,
            password=hasher.hash(user_data.password)
        )
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.put("/", response_model=UserResponse)
def update_password(
    old_password: str,
    new_password: str, 
    current_user: Annotated[User, Depends(get_current_user)],
    session: SessionDep
):
    try:
        auth_user = authenticate_user(session, current_user.username, old_password)
        if not auth_user:
            raise HTTPException(status_code=404, detail="password does not match")
        
        current_user.hashed_password = hasher.hash(new_password)
        session.add(current_user)
        session.commit()
        session.refresh(current_user)

        return current_user
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Get all users
@router.get("/", response_model=List[UserResponse])
def read_all_users(
    session: SessionDep,
    role: Optional[str] = None,
    skip: int = 0,
    limit: Annotated[int, Query(le=100)] = 100,
):
    try:
        all_users = get_all_users(session, skip, limit)
        return all_users

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Get a user by username
@router.get("/{username}", response_model=UserResponse)
def read_user(username: str, session: SessionDep):
    try:
        user = get_user_by_username(session, username)
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        return user

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")

# Delete a user by username
@router.delete("/{username}")
def delete_user_by_username(username: str, session: SessionDep):
    try:
        user = get_user_by_username(session, username)
        if not user:
            raise HTTPException(status_code=404, detail=f"User '{username}' not found")
        delete_user_by_id(session, user.id)
        return {"ok": True}

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
