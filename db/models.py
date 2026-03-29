from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, func
from db.connection import Base

class ArgoFloat(Base):
    __tablename__ = "argo_floats"
    id            = Column(Integer, primary_key=True)
    wmo_id        = Column(String(20), unique=True, nullable=False)
    dac           = Column(String(20))
    institution   = Column(String(100))
    platform_type = Column(String(50))
    created_at    = Column(DateTime, default=func.now())

class Profile(Base):
    __tablename__ = "profiles"
    id              = Column(Integer, primary_key=True)
    float_id        = Column(Integer, ForeignKey("argo_floats.id"), nullable=False)
    cycle_number    = Column(Integer)
    profile_date    = Column(DateTime)
    latitude        = Column(Float)
    longitude       = Column(Float)
    direction       = Column(String(1))
    profile_pres_qc = Column(String(1))
    source_file     = Column(String(300))
    created_at      = Column(DateTime, default=func.now())

class Measurement(Base):
    __tablename__ = "measurements"
    id          = Column(Integer, primary_key=True)
    profile_id  = Column(Integer, ForeignKey("profiles.id"), nullable=False)
    depth_level = Column(Integer)
    pressure    = Column(Float)
    pres_qc     = Column(String(1))
    temperature = Column(Float)
    temp_qc     = Column(String(1))
    salinity    = Column(Float)
    sal_qc      = Column(String(1))
    created_at  = Column(DateTime, default=func.now())