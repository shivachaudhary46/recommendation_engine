from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Annotated, Optional

router = APIRouter(
    prefix="/users",
    tags=["users"]
)

from app.models.models import User
from app.vector_db.database import SessionDep
from app.api.crud import get_user_by_email


@router.post("/")
def create_new_user(user: User, session: SessionDep):
    try: 
        existing = get_user_by_email(session, user.email)
        if existing: 
            raise HTTPException(status_code=400, detail="email already exists")
        
        session.add(user)
        session.commit()
        session.refresh(user)
        return user
    
    except HTTPException:
        raise

    except Exception as e: 
        raise HTTPException(status_code=500, detail="Internal Server Error")        
