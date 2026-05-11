from sqlalchemy import Column, BigInteger, SmallInteger, String, Numeric, Boolean, Text, DateTime
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

class CageReading(Base):
    __tablename__ = "cage_readings"
    id           = Column(BigInteger, primary_key=True, autoincrement=True)
    cage_id      = Column(String(20), nullable=False, index=True)
    recorded_at  = Column(DateTime(timezone=True), nullable=False, index=True)
    mouse_count  = Column(SmallInteger)
    water_pct    = Column(Numeric(5, 2))
    water_status = Column(String(10))
    food_pct     = Column(Numeric(5, 2))
    food_status  = Column(String(10))
    inference_ms = Column(SmallInteger)
    image_path   = Column(Text)

class Alert(Base):
    __tablename__ = "alerts"
    id           = Column(BigInteger, primary_key=True, autoincrement=True)
    cage_id      = Column(String(20), index=True)
    alert_type   = Column(String(20))
    triggered_at = Column(DateTime(timezone=True))
    resolved_at  = Column(DateTime(timezone=True), nullable=True)
    notified     = Column(Boolean, default=False)