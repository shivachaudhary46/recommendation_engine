from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List
from datetime import datetime
from enum import Enum  

class User(SQLModel, table=True):
    __tablename__ = "users"
    id: int | None = Field(default=None, primary_key=True)
    full_name: str
    email: str = Field(index=True, unique=True)
    created_at: datetime = Field(default_factory=datetime.now)

    reactions: List["Reaction"] = Relationship(back_Populates="user")
    interactions: List["Interaction"] = Relationship(back_populates="user")
    Watchlist: List["Watchlist"] = Relationship(back_populates="user")

class Movie(SQLModel, table=True):
    __tablename__ = "movies"
    id: int | None = Field(default=None, primary_key=True)
    title: str = Field(index=True)
    genre: str
    description: str
    imdb_rating: Optional[float]
    release_year: int
    duration_minutes: int
    content_type: str
    created_at: datetime = Field(default_factory=datetime.now)

    reactions: List["Reaction"] = Relationship(back_populates="movie")
    interactions: List["Interaction"] = Relationship(back_populates="movie")
    watchlist: List["Watchlist"] = Relationship(back_populates="movie")

class Reaction(SQLModel, table=True):
    __tablename__ = "reactions"
    id: int | None = Field(default=None, primary_key=True)
    score: List[str]
    user_id: Optional[int] = Field(default=None, foreign_key="users.id", index=True)
    movie_id: Optional[int] = Field(default=None, foreign_key="movies.id", index=True)
    emoji_type: str 
    emoji_score: float
    created_at: datetime = Field(default_factory=datetime.now)

    user: Optional["User"] = Relationship(back_populates="reactions")
    movie: Optional["Movie"] = Relationship(back_populates="reactions")

class Watchlist(SQLModel, table=True):
    __tablename__ = "watchlists"
    id: int | None = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="users.id", index=True)
    movie_id: Optional[int] = Field(default=None, foreign_key="movies.id", index=True)
    added_at: datetime = Field(default_factory=datetime.now)

    user: Optional["User"] = Relationship(back_populates="watchlist")
    movie: Optional["Movie"] = Relationship(back_populates="watchlist")
    
class Interaction(SQLModel, table=True):
    __tablename__ = "interactions"
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: Optional[int] = Field(default=None, foreign_key="users.id", index=True)
    movie_id: Optional[int] = Field(default=None, foreign_key="movies.id", index=True)
    
    action_type: str # view, click, watch, dislike 
    weight: float
    created_at: datetime = Field(default_factory=datetime.now)
    feedback: str

    user: Optional["User"] = Relationship(back_populates="interactions")
    movie: Optional["Movie"] = Relationship(back_populates="interactions")


