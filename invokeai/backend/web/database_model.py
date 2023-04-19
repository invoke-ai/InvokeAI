import ssl

from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, ForeignKey, String, Text, DateTime, SmallInteger, Float, func, Integer, VARCHAR
from pydantic import BaseSettings


class DBConfig(BaseSettings):
    # Database configs
    dialect: str
    user: str
    password: str
    server: str
    database: str
    certificate: str = "/certificates/db_certificate.crt.pem"

    class Config:
        env_prefix = "db_"
        case_sensitive = True


DB_CONFIG = DBConfig()
DATABASE_URL = f"{DB_CONFIG.dialect}://{DB_CONFIG.user}:{DB_CONFIG.password}@{DB_CONFIG.server}/{DB_CONFIG.database}"
Base = declarative_base()


class Parameters(Base):
    # Table name
    __tablename__ = 'parameters'

    # Table columns
    id = Column(Integer, unique=True, nullable=False, primary_key=True, autoincrement=True)
    prompt = Column(Text, unique=False, nullable=True)
    negative_prompt = Column(Text, unique=False, nullable=True)
    image_height = Column(SmallInteger, unique=False, nullable=False)
    image_width = Column(SmallInteger, unique=False, nullable=False)
    guidance = Column(Float, unique=False, nullable=False)
    denoising_strength = Column(Float, unique=False, nullable=False)
    num_of_image = Column(SmallInteger, unique=False, nullable=False)
    created_at = Column(DateTime, unique=False, nullable=False, default=func.now())

    # Table relationships
    job_id = Column(Integer, ForeignKey('jobs.id'))
    job = relationship('Job', back_populates='parameters', cascade='all, delete')


class InputImage(Base):
    # Table name
    __tablename__ = 'input_image'

    # Table columns
    id = Column(Integer, unique=True, nullable=False, primary_key=True, autoincrement=True)
    img_path = Column(VARCHAR(length=255), unique=True, nullable=False)
    created_at = Column(DateTime, unique=False, nullable=False, default=func.now())

    # Table relationships
    job_id = Column(Integer, ForeignKey('jobs.id'))

    # processed_image_id = Column(Integer, ForeignKey('processed_image.id'))
    job = relationship('Job', back_populates='input_image', cascade='all, delete')
    processed_image = relationship('ProcessedImage', back_populates='input_image', uselist=False, cascade='all, delete')


class ProcessedImage(Base):
    # Table name
    __tablename__ = 'processed_image'

    # Table columns
    id = Column(Integer, unique=True, nullable=False, primary_key=True, autoincrement=True)
    img_path = Column(VARCHAR(length=255), unique=True, nullable=False)
    created_at = Column(DateTime, unique=False, nullable=False, default=func.now())

    # Table relationships
    job_id = Column(Integer, ForeignKey('jobs.id'))
    input_image_id = Column(Integer, ForeignKey('input_image.id'))

    job = relationship('Job', back_populates='processed_image', cascade='all, delete')
    input_image = relationship('InputImage', back_populates='processed_image', uselist=False, cascade='all, delete')


class GeneratedImage(Base):
    # Table name
    __tablename__ = 'generated_image'

    # Table columns
    id = Column(Integer, unique=True, nullable=False, primary_key=True, autoincrement=True)
    img_path = Column(VARCHAR(length=255), unique=True, nullable=False)
    created_at = Column(DateTime, unique=False, nullable=False, default=func.now())

    # Table relationships
    job_id = Column(Integer, ForeignKey('jobs.id'))
    job = relationship('Job', back_populates='generated_image', cascade='all, delete')
    feedback = relationship('Feedback', back_populates='generated_image', cascade='all, delete')


class Feedback(Base):
    # Table name
    __tablename__ = 'feedback'
    
    # Table columns
    id = Column(Integer, unique=True, nullable=False, primary_key=True, autoincrement=True)
    feedback = Column(String(5), unique=False, nullable=False)
    feedback_text = Column(Text, unique=False, nullable=False)

    # Table relationships
    generated_image_id = Column(Integer, ForeignKey('generated_image.id'))
    generated_image = relationship('GeneratedImage', back_populates='feedback')


class Job(Base):
    # Table name
    __tablename__ = 'jobs'

    # Table columns
    id = Column(Integer, unique=True, nullable=False, primary_key=True, autoincrement=True)

    # Table relationships
    parameters = relationship('Parameters', uselist=False, back_populates='job')
    input_image = relationship('InputImage', uselist=True, back_populates='job')
    processed_image = relationship('ProcessedImage', uselist=True, back_populates='job')
    generated_image = relationship('GeneratedImage', uselist=True, back_populates='job')



# Set SSL connection argument for DB
ctx = ssl.SSLContext()
ctx.check_hostname = False
ctx.load_verify_locations(cafile=DB_CONFIG.certificate)
ssl_args = {"ssl": ctx}


# Create engin and sessionmaker according to database connection string and SSL information
engine = create_engine(DATABASE_URL, connect_args=ssl_args)
Base.metadata.create_all(engine)
session_maker = sessionmaker(bind=engine)
