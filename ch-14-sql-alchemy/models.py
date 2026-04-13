from __future__ import annotations
from datetime import UTC,datetime

from sqlalchemy import Column, DateTime, Integer, String,Text,ForeignKey
from sqlalchemy.orm import relationship,mapped_column,Mapped

from database import Base

class User(Base):
    __tablename__ = "users" 
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    image_file: Mapped[str | None] = mapped_column(String(200), nullable=True, default=None)
    posts = Mapped[list[Post]] = relationship("Post", back_populates="author")

    @property
    def image_path(self) -> str:
        if self.image_file:
            return f"/media/profile_pics/{self.image_file}"
        return "/static/profile_pics/default.jpg"
    