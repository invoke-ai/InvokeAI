import { useMemo } from "react"
import { useListAllBoardsQuery } from "services/api/endpoints/boards";
import { useListImagesQuery } from "services/api/endpoints/images";

export const useHasImages = () => {
    const { data: boardList } = useListAllBoardsQuery({ include_archived: true })
    const { data: uncategorizedImages } = useListImagesQuery({
        board_id: "none",
        offset: 0,
        limit: 0,
        is_intermediate: false,
    })

    const hasImages = useMemo(() => {
        const hasBoards = boardList && boardList.length > 0;

        if (hasBoards) {
            if (boardList.filter(board => board.image_count > 0).length > 0) {
                return true
            }
        }
        return uncategorizedImages ? uncategorizedImages.total > 0 : false


    }, [boardList, uncategorizedImages])

    return hasImages
}