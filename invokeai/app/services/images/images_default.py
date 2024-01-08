from typing import Optional, List, Tuple

import time
from datetime import datetime

from PIL import Image
from PIL.Image import Image as PILImageType

from invokeai.app.invocations.baseinvocation import MetadataField
from invokeai.app.services.invoker import Invoker
from invokeai.app.services.shared.pagination import OffsetPaginatedResults
from invokeai.app.services.workflow_records.workflow_records_common import WorkflowWithoutID

from ..image_files.image_files_common import (
    ImageFileDeleteException,
    ImageFileNotFoundException,
    ImageFileSaveException,
)
from ..image_records.image_records_common import (
    ImageCategory,
    ImageRecord,
    ImageRecordChanges,
    ImageRecordDeleteException,
    ImageRecordNotFoundException,
    ImageRecordSaveException,
    InvalidImageCategoryException,
    InvalidOriginException,
    ResourceOrigin,
)
from .images_base import ImageServiceABC
from .images_common import ImageDTO, image_record_to_dto, ImageUploadData


class ImageService(ImageServiceABC):
    __invoker: Invoker

    def start(self, invoker: Invoker) -> None:
        self.__invoker = invoker

    ############################################################################################################
    ############################    Eryx CODE   ################################################################
    ############################################################################################################
    """ The create_eryx method is the new create method that will be used to upload images to the system.
    This is the object it receives for 1,2,....n images:
    Images: [
        ImageUploadData(
            image=<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=5616x3744 at 0x7FBC96A07AD0>,
            image_origin=<ResourceOrigin.EXTERNAL: 'external'>,
            image_category=<ImageCategory.USER: 'user'>,
            session_id=None,
            board_id=None,
            is_intermediate=False,
            metadata=None,
            workflow=None,
            size=None
        ),
        ImageUploadData(
            image=<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=275x183 at 0x7FBC9A6A1390>,
            image_origin=<ResourceOrigin.EXTERNAL: 'external'>,
            image_category=<ImageCategory.USER: 'user'>,
            session_id=None,
            board_id=None,
            is_intermediate=False,
            metadata=None,
            workflow=None,
            size=None
        )
    ]

    This is the object of ImageDTOs it returns:
    Image DTOs: [
        ImageUploadData(
            image=<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1200x1197 at 0x7F8127133110>, 
            image_name='e7808a29-0faa-43b4-97c6-bfe86989d2d1.png', 
            image_origin=<ResourceOrigin.EXTERNAL: 'external'>, 
            image_category=<ImageCategory.USER: 'user'>, 
            session_id=None, 
            board_id=None, 
            is_intermediate=False, 
            metadata=None, 
            workflow=None, 
            size=None
        ), 
        ImageUploadData(
            image=<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1200x1197 at 0x7F8127124710>, 
            image_name='f776669b-4e24-45a4-b430-32c6a4ba1f38.png', 
            image_origin=<ResourceOrigin.EXTERNAL: 'external'>, 
            image_category=<ImageCategory.USER: 'user'>, 
            session_id=None, 
            board_id=None, 
            is_intermediate=False, 
            metadata=None, 
            workflow=None, 
            size=None
        )
    ]
    """
    # this the images_default create function
    # TODO: Add additional processing from the process_images method
    # TODO: Add image record code here for multiple insert as well
    # TODO: change the return list to image DTOs from None
    async def create_eryx(self, upload_data_list: List[ImageUploadData]) -> List[ImageDTO]:
        """Starts the upload process"""
        # TODO: Implement the start upload process method here
        print("-----------------------------------")
        print(f"{datetime.now()}")
        print("Starting upload process")
        print("-----------------------------------")
        print(f"Images: {upload_data_list}")
        print("-----------------------------------")
        
        images_DTOs = []
        
        # TODO: Add additional processing here
        for idx, image in enumerate(upload_data_list):
            if image.image_origin not in ResourceOrigin:
                raise InvalidOriginException

            if image.image_category not in ImageCategory:
                raise InvalidImageCategoryException
            
            # Extract width and height from the PIL image and add it to image DTO
            width, height = image.image.size
            image.width = width
            image.height = height

            # Create a name for the image and add it to image DTO
            image_name = self.__invoker.services.names.create_image_name()
            image.image_name = image_name

            # append the image to the list of images DTOs
            images_DTOs.append(image)
            try:
                # TODO: Consider using a transaction here to ensure consistency between storage and database
                # TODO: Change arguments to just one images_DTOs[idx] instead of all the arguments
                self.__invoker.services.image_records.save_record_eryx(images_DTOs)

                # Link image to board if board_id is provided
                if image.board_id is not None:
                    self.__invoker.services.board_image_records.add_image_to_board(board_id=image.board_id, image_name=image.image_name)
                
                # Save the image file
                self.__invoker.services.image_files.save(
                    image_name=image.image_name, image=image.image, metadata=image.metadata, workflow=image.workflow
                )

                # Retrieve the created ImageDTO
                image_dto = self.get_dto(image_name)
                # Replace in place the image data to ImageDTO created
                images_DTOs[idx] = image_dto

                print("Image DTO was created here <------------------")
                print(f"Image DTO: {image_dto}")
                print(f"this is the image url: {image_dto.image_url}")
                self._on_changed(image_dto)
                print("This is after the on_changed method was called <------------------")
                print(f"{image_dto}")
                print("-----------------------------------")

            except ImageRecordSaveException:
                self.__invoker.services.logger.error("Failed to save image record")
                raise
            except ImageFileSaveException:
                self.__invoker.services.logger.error("Failed to save image file")
                raise
            except Exception as e:
                self.__invoker.services.logger.error(f"Problem saving image record and file: {str(e)}")
                raise e


        print(f"Image DTOs: {images_DTOs}")
        return images_DTOs

    ############################################################################################################
    ############################    ORIGINAL CODE   ############################################################
    ############################################################################################################
    def create(
        self,
        image: PILImageType,
        image_origin: ResourceOrigin,
        image_category: ImageCategory,
        node_id: Optional[str] = None,
        session_id: Optional[str] = None,
        board_id: Optional[str] = None,
        is_intermediate: Optional[bool] = False,
        metadata: Optional[MetadataField] = None,
        workflow: Optional[WorkflowWithoutID] = None,
    ) -> ImageDTO:
        if image_origin not in ResourceOrigin:
            raise InvalidOriginException

        if image_category not in ImageCategory:
            raise InvalidImageCategoryException

        image_name = self.__invoker.services.names.create_image_name()

        (width, height) = image.size

        try:
            # TODO: Consider using a transaction here to ensure consistency between storage and database
            self.__invoker.services.image_records.save(
                # Non-nullable fields
                image_name=image_name,
                image_origin=image_origin,
                image_category=image_category,
                width=width,
                height=height,
                has_workflow=workflow is not None,
                # Meta fields
                is_intermediate=is_intermediate,
                # Nullable fields
                node_id=node_id,
                metadata=metadata,
                session_id=session_id,
            )
            if board_id is not None:
                self.__invoker.services.board_image_records.add_image_to_board(board_id=board_id, image_name=image_name)
                
            self.__invoker.services.image_files.save(
                image_name=image_name, image=image, metadata=metadata, workflow=workflow
            )
            image_dto = self.get_dto(image_name)

            self._on_changed(image_dto)
            return image_dto
        except ImageRecordSaveException:
            self.__invoker.services.logger.error("Failed to save image record")
            raise
        except ImageFileSaveException:
            self.__invoker.services.logger.error("Failed to save image file")
            raise
        except Exception as e:
            self.__invoker.services.logger.error(f"Problem saving image record and file: {str(e)}")
            raise e
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################

    def update(
        self,
        image_name: str,
        changes: ImageRecordChanges,
    ) -> ImageDTO:
        try:
            self.__invoker.services.image_records.update(image_name, changes)
            image_dto = self.get_dto(image_name)
            self._on_changed(image_dto)
            return image_dto
        except ImageRecordSaveException:
            self.__invoker.services.logger.error("Failed to update image record")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem updating image record")
            raise e

    def get_pil_image(self, image_name: str) -> PILImageType:
        try:
            return self.__invoker.services.image_files.get(image_name)
        except ImageFileNotFoundException:
            self.__invoker.services.logger.error("Failed to get image file")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image file")
            raise e

    def get_record(self, image_name: str) -> ImageRecord:
        try:
            return self.__invoker.services.image_records.get(image_name)
        except ImageRecordNotFoundException:
            self.__invoker.services.logger.error("Image record not found")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image record")
            raise e

    def get_dto(self, image_name: str) -> ImageDTO:
        try:
            image_record = self.__invoker.services.image_records.get(image_name)

            image_dto = image_record_to_dto(
                image_record=image_record,
                image_url=self.__invoker.services.urls.get_image_url(image_name),
                thumbnail_url=self.__invoker.services.urls.get_image_url(image_name, True),
                board_id=self.__invoker.services.board_image_records.get_board_for_image(image_name),
            )
            
            return image_dto
        except ImageRecordNotFoundException:
            self.__invoker.services.logger.error("Image record not found")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image DTO")
            raise e

    def get_metadata(self, image_name: str) -> Optional[MetadataField]:
        try:
            return self.__invoker.services.image_records.get_metadata(image_name)
        except ImageRecordNotFoundException:
            self.__invoker.services.logger.error("Image record not found")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image DTO")
            raise e

    def get_workflow(self, image_name: str) -> Optional[WorkflowWithoutID]:
        try:
            return self.__invoker.services.image_files.get_workflow(image_name)
        except ImageFileNotFoundException:
            self.__invoker.services.logger.error("Image file not found")
            raise
        except Exception:
            self.__invoker.services.logger.error("Problem getting image workflow")
            raise

    def get_path(self, image_name: str, thumbnail: bool = False) -> str:
        try:
            return str(self.__invoker.services.image_files.get_path(image_name, thumbnail))
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image path")
            raise e

    def validate_path(self, path: str) -> bool:
        try:
            return self.__invoker.services.image_files.validate_path(path)
        except Exception as e:
            self.__invoker.services.logger.error("Problem validating image path")
            raise e

    def get_url(self, image_name: str, thumbnail: bool = False) -> str:
        try:
            return self.__invoker.services.urls.get_image_url(image_name, thumbnail)
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting image path")
            raise e

    def get_many(
        self,
        offset: int = 0,
        limit: int = 10,
        image_origin: Optional[ResourceOrigin] = None,
        categories: Optional[list[ImageCategory]] = None,
        is_intermediate: Optional[bool] = None,
        board_id: Optional[str] = None,
    ) -> OffsetPaginatedResults[ImageDTO]:
        try:
            results = self.__invoker.services.image_records.get_many(
                offset,
                limit,
                image_origin,
                categories,
                is_intermediate,
                board_id,
            )

            image_dtos = [
                image_record_to_dto(
                    image_record=r,
                    image_url=self.__invoker.services.urls.get_image_url(r.image_name),
                    thumbnail_url=self.__invoker.services.urls.get_image_url(r.image_name, True),
                    board_id=self.__invoker.services.board_image_records.get_board_for_image(r.image_name),
                )
                for r in results.items
            ]

            return OffsetPaginatedResults[ImageDTO](
                items=image_dtos,
                offset=results.offset,
                limit=results.limit,
                total=results.total,
            )
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting paginated image DTOs")
            raise e

    def delete(self, image_name: str):
        try:
            self.__invoker.services.image_files.delete(image_name)
            self.__invoker.services.image_records.delete(image_name)
            self._on_deleted(image_name)
        except ImageRecordDeleteException:
            self.__invoker.services.logger.error("Failed to delete image record")
            raise
        except ImageFileDeleteException:
            self.__invoker.services.logger.error("Failed to delete image file")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem deleting image record and file")
            raise e

    def delete_images_on_board(self, board_id: str):
        try:
            image_names = self.__invoker.services.board_image_records.get_all_board_image_names_for_board(board_id)
            for image_name in image_names:
                self.__invoker.services.image_files.delete(image_name)
            self.__invoker.services.image_records.delete_many(image_names)
            for image_name in image_names:
                self._on_deleted(image_name)
        except ImageRecordDeleteException:
            self.__invoker.services.logger.error("Failed to delete image records")
            raise
        except ImageFileDeleteException:
            self.__invoker.services.logger.error("Failed to delete image files")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem deleting image records and files")
            raise e

    def delete_intermediates(self) -> int:
        try:
            image_names = self.__invoker.services.image_records.delete_intermediates()
            count = len(image_names)
            for image_name in image_names:
                self.__invoker.services.image_files.delete(image_name)
                self._on_deleted(image_name)
            return count
        except ImageRecordDeleteException:
            self.__invoker.services.logger.error("Failed to delete image records")
            raise
        except ImageFileDeleteException:
            self.__invoker.services.logger.error("Failed to delete image files")
            raise
        except Exception as e:
            self.__invoker.services.logger.error("Problem deleting image records and files")
            raise e

    def get_intermediates_count(self) -> int:
        try:
            return self.__invoker.services.image_records.get_intermediates_count()
        except Exception as e:
            self.__invoker.services.logger.error("Problem getting intermediates count")
            raise e
