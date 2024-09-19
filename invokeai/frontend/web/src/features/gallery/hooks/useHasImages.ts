import { selectListImagesQueryArgs } from "features/gallery/store/gallerySelectors";
import { useCallback } from "react"
import { useListAllBoardsQuery } from "services/api/endpoints/boards";
import { useListImagesQuery } from "services/api/endpoints/images";

export const useHasImages = () => {
    const { data: boardList } = useListAllBoardsQuery({ include_archived: true })
    const { data: uncategorizedImages } = useListImagesQuery({ ...selectListImagesQueryArgs, board_id: "none" })

    const hasImages = useCallback(() => {
        const hasBoards = boardList && boardList.length > 0;
        if (!hasBoards) {
            return uncategorizedImages ? uncategorizedImages.total > 0 : false
        } else {
            return boardList.filter(board => board.image_count > 0).length > 0
        }
    }, [boardList, uncategorizedImages])

    return hasImages
}