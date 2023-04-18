import uuid
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, ForeignKey, String, Text, DateTime, SmallInteger, Float, func, UUID


Base = declarative_base()


class Parameters(Base):
    # Table name
    __tablename__ = 'parameters'

    # Table columns
    id = Column(UUID(as_uuid=True), unique=True, nullable=False, primary_key=True, default=uuid.uuid4)
    prompt = Column(Text, unique=False, nullable=True)
    negative_prompt = Column(Text, unique=False, nullable=True)
    image_height = Column(SmallInteger(unsigned=True), unique=False, nullable=False)
    image_width = Column(SmallInteger(unsigned=True), unique=False, nullable=False)
    guidance = Column(Float, unique=False, nullable=False)
    denoising_strength = Column(Float, unique=False, nullable=False)
    num_of_image = Column(SmallInteger(unsigned=True), unique=False, nullable=False)
    created_at = Column(DateTime, unique=False, nullable=False, default=func.now())

    # Table relationships
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.id'))
    job = relationship('Job', back_populates='parameters', cascade='all, delete')


class InputImage(Base):
    # Tabel name
    __tablename__ = 'input_image'

    # Table columns
    id = Column(UUID(as_uuid=True), unique=True, nullable=False, primary_key=True, default=uuid.uuid4)
    img_path = Column(Text, unique=True, nullable=False)
    created_at = Column(DateTime, unique=False, nullable=False, default=func.now())

    # Table relationships
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.id'))
    job = relationship('Job', back_populates='input_image', cascade='all, delete')


class ProcessedImage(Base):
    # Table name
    __tablename__ = 'processed_image'

    # Table columns
    id = Column(UUID(as_uuid=True), unique=True, nullable=False, primary_key=True, default=uuid.uuid4)
    img_path = Column(Text, unique=True, nullable=False)
    created_at = Column(DateTime, unique=False, nullable=False, default=func.now())

    # Table relationships
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.id'))
    input_image_id = Column(UUID(as_uuid=True), ForeignKey('input_image.id'))
    job = relationship('Job', back_populates='processed_image', cascade='all, delete')
    input_image = relationship('InputImage', back_populates='processed_image', cascade='all, delete')


class GeneratedImage(Base):
    # Tabel name
    __tablename__ = 'generated_image'

    # Tabel columns
    id = Column(UUID(as_uuid=True), unique=True, nullable=False, primary_key=True, default=uuid.uuid4)
    img_path = Column(Text, unique=True, nullable=False)
    created_at = Column(DateTime, unique=False, nullable=False, default=func.now())

    # Table relationships
    job_id = Column(UUID(as_uuid=True), ForeignKey('jobs.id'))
    job = relationship('Job', back_populates='generated_image', cascade='all, delete')
    feedback = relationship('Feedback', back_populates='generated_image', cascade='all, delete')


class Feedback(Base):
    # Tabel name
    __tablename__ = 'feedback'
    
    # Tabel columns
    id = Column(UUID(as_uuid=True), unique=True, nullable=False, primary_key=True, default=uuid.uuid4)
    feedback = Column(String(5), unique=False, nullable=False)
    feedback_text = Column(Text, unique=False, nullable=False)

    # Table relationships
    generated_image_id = Column(UUID(as_uuid=True), ForeignKey('generated_image.id'))
    generated_image = relationship('GeneratedImage', back_populates='feedback')


class Job(Base):
    # Tabel name
    __tablename__ = 'jobs'

    # Tabel columns
    id = Column(UUID(as_uuid=True), unique=True, nullable=False, primary_key=True, default=uuid.uuid4)

    # Table relationships
    parameters = relationship('Parameters', uselist=False, back_populates='job')
    input_image = relationship('InputImage', back_populates='job')
    processed_image = relationship('ProcessedImage', back_populates='job')
    generated_image = relationship('GeneratedImage', back_populates='job')
